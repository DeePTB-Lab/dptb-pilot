import os
import re
import httpx
from dptb_pilot.tools.init import mcp
from typing import List, Dict, Any
from pymatgen.core import Structure

# COD API 基础 URL
COD_BASE_URL = "https://www.crystallography.net/cod"

@mcp.tool()
def search_cod_structures(query: str, limit: int = 5) -> str:
    """
    Search for crystal structures in the Crystallography Open Database (COD).

    Args:
        query: Search query (chemical formula, element name, etc.).
        limit: Maximum number of results to return (default: 5).

    Returns:
        Formatted list of materials with COD IDs and properties.
    """
    try:
        # COD 提供了多种搜索方式
        # 方法1: 使用 COD 的文本搜索接口
        # 方法2: 使用简单的化学式搜索
        # 让我们尝试一种更通用的方法

        # 首先尝试通过化学式搜索
        if re.match(r'^[A-Za-z0-9\s\(\)]+$', query) and not query.startswith('COD'):
            # 尝试作为化学式搜索
            formula_url = f"{COD_BASE_URL}/cod/?formula={query}&format=json&limit={limit}"
        else:
            # 通用文本搜索
            search_url = f"{COD_BASE_URL}/search?q={query}&format=json&limit={limit}"
            formula_url = None

        results = []

        with httpx.Client(timeout=15.0) as client:
            # 首先尝试化学式搜索（如果适用）
            if formula_url:
                try:
                    response = client.get(formula_url)
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, dict) and "data" in data:
                            results.extend(data["data"])
                        elif isinstance(data, list):
                            results.extend(data)
                except:
                    pass  # 忽略这个方法的错误，继续尝试其他方法

            # 如果没有结果或不是化学式查询，尝试其他搜索方式
            if not results:
                # 尝试 COD 的 OPTIMADE API 端点
                try:
                    optimade_url = f"https://optimade.crystallography.net/v1/structures"
                    params = {
                        "filter": f'chemical_formula_descriptive CONTAINS "{query}"',
                        "page_limit": str(limit),
                        "response_fields": "id,chemical_formula_descriptive,space_group_symbol_standard,cell_length_a,cell_length_b,cell_length_c,cell_volume"
                    }
                    response = client.get(optimade_url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data:
                            for item in data["data"]:
                                results.append({
                                    "id": item["id"].split("-")[-1],  # 提取 COD ID
                                    "formula": item["attributes"].get("chemical_formula_descriptive", "Unknown"),
                                    "space_group": item["attributes"].get("space_group_symbol_standard", "Unknown"),
                                    "a": item["attributes"].get("cell_length_a", "Unknown"),
                                    "b": item["attributes"].get("cell_length_b", "Unknown"),
                                    "c": item["attributes"].get("cell_length_c", "Unknown"),
                                    "cell_volume": item["attributes"].get("cell_volume", "Unknown")
                                })
                except:
                    pass

        if not results:
            return f"No materials found in COD for query: {query}"

        # 限制结果数量
        results = results[:limit]

        output = f"Found {len(results)} materials in COD for '{query}' (showing top {len(results)}):\n\n"

        for i, item in enumerate(results):
            cod_id = item.get("id", "Unknown")
            formula = item.get("formula", "Unknown")
            space_group = item.get("space_group", "Unknown")
            cell_volume = item.get("cell_volume", "Unknown")
            a = item.get("a", "Unknown")
            b = item.get("b", "Unknown")
            c = item.get("c", "Unknown")

            output += f"{i+1}. **ID: {cod_id}** - {formula}\n"
            if space_group != "Unknown":
                output += f"   - Space Group: {space_group}\n"
            if a != "Unknown" and b != "Unknown" and c != "Unknown":
                output += f"   - Cell parameters: a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å\n"
            if cell_volume != "Unknown":
                output += f"   - Cell Volume: {cell_volume:.1f} Ų\n"
            output += "\n"

        output += "To download a structure, use the `download_cod_structure` tool with the COD ID."
        return output

    except httpx.RequestError as e:
        return f"Network error accessing COD: {str(e)}"
    except Exception as e:
        return f"Error searching COD: {str(e)}"

@mcp.tool()
def download_cod_structure(cod_id: str, work_path: str = ".") -> str:
    """
    Download a crystal structure from COD by its ID.

    Args:
        cod_id: The COD ID (e.g., "1000001").
        work_path: Directory to save the file (default: current directory).

    Returns:
        Success message with the path to the downloaded file.
    """
    try:
        # COD CIF 下载 URL
        # COD 使用简单的 URL 模式: /cod/{cod_id}.cif
        cif_url = f"{COD_BASE_URL}/{cod_id}.cif"

        with httpx.Client(timeout=30.0) as client:
            response = client.get(cif_url)

            if response.status_code == 404:
                return f"Error: COD ID {cod_id} not found."

            response.raise_for_status()

            # 检查返回的内容是否是有效的 CIF 文件
            cif_content = response.text
            if not cif_content.strip().startswith("data_"):
                return f"Error: Invalid CIF format returned for COD ID {cod_id}."

            # 使用 pymatgen 解析 CIF 内容
            try:
                # 从 CIF 内容解析结构
                from pymatgen.io.cif import CifParser
                from io import StringIO

                cif_parser = CifParser(StringIO(cif_content))
                structures = cif_parser.get_structures()

                if not structures:
                    return f"Error: No valid structure found in CIF for COD ID {cod_id}."

                # 使用第一个结构
                structure = structures[0]

            except Exception as parse_error:
                # 如果 pymatgen 解析失败，仍然保存原始 CIF 文件
                structure = None
                logger_info = f"Warning: Could not parse CIF with pymatgen: {str(parse_error)}. Saving raw CIF."

            # 确保工作路径存在
            if not os.path.exists(work_path):
                try:
                    os.makedirs(work_path, exist_ok=True)
                except Exception as e:
                    return f"Error creating workspace directory {work_path}: {e}"

            # 生成文件名
            filename = f"COD_{cod_id}.cif"
            save_path = os.path.join(work_path, filename)

            # 保存 CIF 文件
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(cif_content)

            # 如果可以解析结构，也保存其他格式
            if structure:
                # 也可以保存为 POSCAR 格式
                poscar_filename = f"COD_{cod_id}.poscar"
                poscar_path = os.path.join(work_path, poscar_filename)
                structure.to(filename=poscar_path)

                success_msg = f"Successfully downloaded COD structure {cod_id} to `{save_path}` (and POSCAR format to `{poscar_path}`). "
            else:
                success_msg = f"Successfully downloaded COD structure {cod_id} to `{save_path}`. "

            success_msg += "You can now visualize it using `visualize_structure`."
            return success_msg

    except httpx.RequestError as e:
        return f"Network error accessing COD: {str(e)}"
    except Exception as e:
        return f"Error downloading COD structure {cod_id}: {str(e)}"