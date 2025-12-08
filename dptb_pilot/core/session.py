from typing import Any, Union

from google.adk.events import Event
from google.adk.sessions import InMemorySessionService, Session, DatabaseSessionService


def get_event(session: Session, index: int) -> Event:
    return session.events[index]


class AdjustableInMemorySessionService(InMemorySessionService):
    def __init__(self):
        super().__init__()

    async def pop_event(self, session: Session) -> Event:
        second_to_last_event = get_event(session, -2)
        self.__update_session_state(session, second_to_last_event)
        session.last_update_time = second_to_last_event.timestamp
        last_event = session.events.pop()

        storage_session = self.sessions[session.app_name][session.user_id].get(session.id)
        storage_session.events.pop()
        storage_session.last_update_time = second_to_last_event.timestamp
        return last_event

async def pop_event(session_service: Union[InMemorySessionService, DatabaseSessionService], session: Session) -> Event:
    async def from_in_memory(_session_service: InMemorySessionService, _session: Session) -> Event:
        second_to_last_event = get_event(_session, -2)
        update_method = getattr(_session_service, '_BaseSessionService__update_session_state')
        update_method(_session, second_to_last_event)
        _session.last_update_time = second_to_last_event.timestamp
        last_event = _session.events.pop()

        storage_session = _session_service.sessions[_session.app_name][_session.user_id].get(_session.id)
        storage_session.events.pop()
        storage_session.last_update_time = second_to_last_event.timestamp
        return last_event

    if isinstance(session_service, InMemorySessionService):
        return await from_in_memory(session_service, session)
    elif isinstance(session_service, DatabaseSessionService):
        raise "not yet support DatabaseSessionService."
    raise "session_service not yet support。"
