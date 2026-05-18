import logging
import os
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
    Gracefully end the current call by deleting the LiveKit room.
    This sends SIP BYE to the carrier, terminating the PSTN leg.
    """
    await context.wait_for_playout()

    room_name = _room_name(context)

    try:
        async with api.LiveKitAPI() as livekit_api:
            await livekit_api.room.delete_room(
                api.DeleteRoomRequest(room=room_name))
            logger.info("Room %s deleted — SIP call terminated.", room_name)
    except Exception as exc:
        logger.warning("Failed to delete room %s: %s", room_name, exc)
        return "Could not end the call."

    return True

@function_tool
async def transfer_call(context: RunContext, transfer_to: str | None = None):
    """
    Transfer the current SIP call to another phone number.
    Uses outbound trunk dial-in when SIP_OUTBOUND_TRUNK_ID is set (preferred),
    otherwise falls back to SIP REFER.
    """
    userdata = _session_userdata(context)
    if userdata is not None and getattr(userdata, "channel", None) != "sip":
        return "Only SIP calls can be transferred."

    # Resolve destination: explicit arg (only if valid E.164) > config > env
    # LLM sometimes passes descriptive text like "human agent" — ignore non-phone values
    valid_transfer_to = transfer_to if transfer_to and _E164_RE.fullmatch(transfer_to.strip()) else None
    raw_destination = (
        valid_transfer_to
        or _transfer_destination_from_context(context)
        or os.getenv("SIP_TRANSFER_TO", "")
    )
    try:
        destination = _normalize_transfer_destination(raw_destination)
        room_name = _room_name(context)
    except ValueError as exc:
        return str(exc)
    except RuntimeError as exc:
        logger.warning("Could not prepare SIP transfer: %s", exc)
        return "Could not transfer the call."

    await context.wait_for_playout()

    sip_outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID", "")

    if sip_outbound_trunk_id:
        # Preferred: dial destination into the room via outbound trunk,
        # then agent leaves — caller and destination stay bridged.
        from livekit.protocol.sip import CreateSIPParticipantRequest
        from google.protobuf.duration_pb2 import Duration

        # Extract plain phone number for sip_call_to
        sip_call_to = raw_destination.replace("tel:", "").replace("sip:", "").strip()
        ringing_timeout = Duration(seconds=int(os.getenv("SIP_TRANSFER_RING_TIMEOUT_SECONDS", "30")))

        logger.info("Dialing %s into room %s via outbound trunk %s", sip_call_to, room_name, sip_outbound_trunk_id)

        # Step 1: Dial destination into the room
        try:
            livekit_api = api.LiveKitAPI()
            try:
                await livekit_api.sip.create_sip_participant(
                    CreateSIPParticipantRequest(
                        sip_trunk_id=sip_outbound_trunk_id,
                        sip_call_to=sip_call_to,
                        room_name=room_name,
                        participant_identity=f"transfer_{sip_call_to}",
                        participant_name="Transfer Destination",
                        play_ringtone=True,
                        wait_until_answered=True,
                        ringing_timeout=ringing_timeout,
                    )
                )
            finally:
                await livekit_api.aclose()
        except Exception as exc:
            logger.warning("Outbound trunk transfer failed: %s", exc)
            return "Could not transfer the call."

        # Step 2: Transfer succeeded — mark as forwarded and agent leaves.
        # Errors here are non-fatal (caller is already connected to destination).
        logger.info("Transfer destination answered. Agent leaving room %s.", room_name)
        if userdata is not None:
            userdata.disconnect_reason = "assistant-forwarded-call"
        try:
            room = getattr(userdata, "room", None) if userdata else None
            if room is not None:
                await room.disconnect()
        except Exception as exc:
            logger.debug("Agent disconnect after transfer (non-fatal): %s", exc)
    else:
        # Fallback: SIP REFER
        try:
            participant_identity = _sip_participant_identity(context)
        except RuntimeError as exc:
            logger.warning("Could not resolve SIP participant: %s", exc)
            return "Could not transfer the call."

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
                    "SIP REFER transfer failed: status_code=%s status=%s",
                    metadata.get("sip_status_code"),
                    metadata.get("sip_status"),
                )
            else:
                logger.warning("SIP REFER transfer failed: %s", exc)
            return "Could not transfer the call."

    logger.info("Transferred call in room %s to %s", room_name, raw_destination)
    return True
