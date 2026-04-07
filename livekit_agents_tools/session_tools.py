import logging
import re
from typing import Any

from livekit import api
from livekit.agents import function_tool, RunContext
from livekit.protocol.sip import TransferSIPParticipantRequest


logger = logging.getLogger("livekit-session-tools")
_E164_RE = re.compile(r"\+[1-9]\d{1,14}$")


def _session_userdata(context: RunContext) -> Any:
    return getattr(context.session, "userdata", None)


def _room_name(context: RunContext) -> str:
    userdata = _session_userdata(context)
    if userdata is not None and getattr(userdata, "room_name", None):
        return userdata.room_name

    # Backwards-compatible fallback for any older runtime that attached JobContext.
    ctx = getattr(userdata, "ctx", None)
    room = getattr(ctx, "room", None)
    if room is not None and getattr(room, "name", None):
        return room.name

    raise RuntimeError("Could not resolve the LiveKit room name for this call.")


def _sip_participant_identity(context: RunContext) -> str:
    userdata = _session_userdata(context)
    if userdata is not None and getattr(userdata, "sip_participant_identity", None):
        return userdata.sip_participant_identity

    ctx = getattr(userdata, "ctx", None)
    room = getattr(ctx, "room", None)
    remote_participants = getattr(room, "remote_participants", {}) or {}
    for participant in remote_participants.values():
        if str(getattr(participant, "kind", "")).lower().endswith("sip"):
            return participant.identity

    raise RuntimeError("Could not resolve the SIP participant identity for this call.")


def _normalize_transfer_destination(transfer_to: str) -> str:
    destination = transfer_to.strip()
    if not destination:
        raise ValueError("Transfer destination phone number is required.")

    if destination.startswith("sip:"):
        return destination

    if destination.startswith("tel:"):
        phone_number = destination.removeprefix("tel:")
    else:
        phone_number = destination

    if not _E164_RE.fullmatch(phone_number):
        raise ValueError("Transfer phone number must be E.164, for example +14155550123.")

    return f"tel:{phone_number}"


def _transfer_destination_from_context(context: RunContext) -> str | None:
    userdata = _session_userdata(context)
    if userdata is None:
        return None
    return getattr(userdata, "transfer_phone_number", None)


@function_tool    
async def end_call(context: RunContext):    
    """    
    Use this function to end the call
    """   
    # Wait for the agent's speech to complete before ending the call  
    await context.wait_for_playout()  
      
    room_name = _room_name(context)
  
    async with api.LiveKitAPI() as livekit_api:  
        await livekit_api.room.delete_room(  
            api.DeleteRoomRequest(room=room_name))  
        print("Room deleted successfully")  
  
    return True

@function_tool    
async def transfer_call(context: RunContext, transfer_to: str | None = None):    
    """    
    Transfer the current SIP call to the UI-provided E.164 phone number using SIP REFER.
    """   
    userdata = _session_userdata(context)
    if userdata is not None and getattr(userdata, "channel", None) != "sip":
        return "Only SIP calls can be transferred."

    try:
        destination = _normalize_transfer_destination(
            transfer_to or _transfer_destination_from_context(context) or ""
        )
        room_name = _room_name(context)
        participant_identity = _sip_participant_identity(context)
    except ValueError as exc:
        return str(exc)
    except RuntimeError as exc:
        logger.warning("Could not prepare SIP transfer: %s", exc)
        return "Could not transfer the call."

    await context.wait_for_playout()  

    try:
        async with api.LiveKitAPI() as livekit_api:
            await livekit_api.sip.transfer_sip_participant(
                TransferSIPParticipantRequest(
                    room_name=room_name,
                    participant_identity=participant_identity,
                    transfer_to=destination,
                    play_dialtone=False,
                )
            )
    except Exception as exc:
        metadata = getattr(exc, "metadata", None) or {}
        if metadata:
            logger.warning(
                "SIP transfer failed: status_code=%s status=%s",
                metadata.get("sip_status_code"),
                metadata.get("sip_status"),
            )
        else:
            logger.warning("SIP transfer failed: %s", exc)
        return "Could not transfer the call."

    logger.info("Transferred SIP participant %s in room %s to %s", participant_identity, room_name, destination)
    return True
