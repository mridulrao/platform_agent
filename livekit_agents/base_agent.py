import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import AsyncIterable, Dict, List, Optional

from livekit import rtc
from livekit.agents import (
    ChatContext,
    ChatMessage,
    FunctionTool,
    JobProcess,
    llm,
    stt,
    tts,
)
from livekit.agents.voice import Agent, ModelSettings, RunContext

logger = logging.getLogger("BaseAgent")

# ---------------------------------------------------------------------------
# Language-specific filler words
# ---------------------------------------------------------------------------
TOOL_CALL_FILLERS: dict[str, list[str]] = {
    "en": [
        "Give me a moment.",
        "Let me check that for you.",
        "One moment please.",
        "Hold on, let me look into that.",
        "Just a second.",
        "Let me find that out.",
        "Bear with me for a moment.",
        "Working on that now.",
    ],
    "hi": [
        "एक पल दीजिए।",
        "मैं अभी देखता हूँ।",
        "बस एक सेकंड।",
        "रुकिए, मैं जाँच करता हूँ।",
        "थोड़ा इंतज़ार कीजिए।",
        "मैं अभी पता करता हूँ।",
        "बस एक मिनट।",
        "अभी देखते हैं।",
    ],
    "fr": [
        "Un instant, s'il vous plaît.",
        "Laissez-moi vérifier.",
        "Je regarde ça tout de suite.",
        "Patientez un moment.",
        "Je m'en occupe.",
        "Donnez-moi un instant.",
        "Je vérifie pour vous.",
        "Un moment, je vous prie.",
    ],
    "es": [
        "Un momento, por favor.",
        "Déjame verificar eso.",
        "Enseguida lo reviso.",
        "Un segundo, por favor.",
        "Permítame un momento.",
        "Voy a comprobarlo.",
        "Deme un instante.",
        "Ya lo reviso.",
    ],
    "it": [
        "Un momento, per favore.",
        "Lasciatemi controllare.",
        "Verifico subito.",
        "Un attimo, prego.",
        "Ci penso io.",
        "Un secondo, per cortesia.",
        "Sto verificando.",
        "Datemi un istante.",
    ],
    "pt": [
        "Um momento, por favor.",
        "Deixe-me verificar.",
        "Vou conferir isso agora.",
        "Só um segundo.",
        "Aguarde um instante.",
        "Estou verificando.",
        "Um instante, por favor.",
        "Vou checar para você.",
    ],
    "de": [
        "Einen Moment bitte.",
        "Lassen Sie mich das prüfen.",
        "Ich schaue mal nach.",
        "Einen Augenblick bitte.",
        "Ich kümmere mich darum.",
        "Nur eine Sekunde.",
        "Ich überprüfe das kurz.",
        "Moment bitte.",
    ],
    "ru": [
        "Одну секунду, пожалуйста.",
        "Сейчас проверю.",
        "Дайте мне минутку.",
        "Подождите немного.",
        "Сейчас посмотрю.",
        "Один момент.",
        "Я сейчас уточню.",
        "Минуточку.",
    ],
    "nl": [
        "Een moment alstublieft.",
        "Ik ga dat even nakijken.",
        "Een ogenblikje.",
        "Ik kijk het even voor u op.",
        "Eén momentje.",
        "Ik controleer het even.",
        "Even geduld alstublieft.",
        "Ik zoek het even uit.",
    ],
    "sv": [
        "Ett ögonblick.",
        "Låt mig kolla det.",
        "En liten stund.",
        "Jag kollar upp det.",
        "Vänta ett ögonblick.",
        "Jag tittar på det nu.",
        "Bara en sekund.",
        "Jag undersöker det.",
    ],
    "uk": [
        "Одну хвилинку.",
        "Зараз перевірю.",
        "Дайте мені мить.",
        "Зачекайте, будь ласка.",
        "Зараз подивлюся.",
        "Один момент.",
        "Я зараз з'ясую.",
        "Хвилиночку.",
    ],
}

FILLER_WORDS: dict[str, list[str]] = {
    "en": [
        "Okay, ",
        "All right, ",
        "Got it, ",
        "Understood, ",
        "Makes sense, ",
        "Sure, ",
        "Right, ",
        "Just a moment, ",
    ],
    "hi": [
        "अच्छा, ",
        "ठीक है, ",
        "समझ गया, ",
        "बिल्कुल, ",
        "एक पल, ",
        "जी, ",
        "सही है, ",
        "बस एक सेकंड, ",
    ],
    "fr": [
        "D'accord, ",
        "Très bien, ",
        "Compris, ",
        "Bien sûr, ",
        "Entendu, ",
        "Un instant, ",
        "C'est noté, ",
        "Parfait, ",
    ],
    "es": [
        "De acuerdo, ",
        "Entendido, ",
        "Muy bien, ",
        "Claro, ",
        "Por supuesto, ",
        "Un momento, ",
        "Perfecto, ",
        "Bien, ",
    ],
    "it": [
        "Va bene, ",
        "Capito, ",
        "D'accordo, ",
        "Certo, ",
        "Un momento, ",
        "Perfetto, ",
        "Benissimo, ",
        "Chiaro, ",
    ],
    "pt": [
        "Certo, ",
        "Entendido, ",
        "Muito bem, ",
        "Claro, ",
        "Um momento, ",
        "Perfeito, ",
        "Com certeza, ",
        "Tudo bem, ",
    ],
    "de": [
        "In Ordnung, ",
        "Verstanden, ",
        "Alles klar, ",
        "Natürlich, ",
        "Einen Moment, ",
        "Gut, ",
        "Genau, ",
        "Selbstverständlich, ",
    ],
    "ru": [
        "Хорошо, ",
        "Понял, ",
        "Конечно, ",
        "Одну секунду, ",
        "Ясно, ",
        "Разумеется, ",
        "Так, ",
        "Момент, ",
    ],
    "nl": [
        "Oké, ",
        "Begrepen, ",
        "Natuurlijk, ",
        "Een moment, ",
        "Goed, ",
        "Prima, ",
        "Zeker, ",
        "Duidelijk, ",
    ],
    "sv": [
        "Okej, ",
        "Förstått, ",
        "Absolut, ",
        "Ett ögonblick, ",
        "Bra, ",
        "Visst, ",
        "Klart, ",
        "Självklart, ",
    ],
    "uk": [
        "Добре, ",
        "Зрозуміло, ",
        "Звичайно, ",
        "Одну мить, ",
        "Так, ",
        "Гаразд, ",
        "Ясно, ",
        "Секундочку, ",
    ],
}


class BaseAgent(Agent):
    def __init__(self, agent_name: str, *args, language: str = "en", **kwargs):
        kwargs.setdefault('allow_interruptions', True)
        super().__init__(*args, **kwargs)

        self.language = language
        self.filler_words = FILLER_WORDS.get(language, FILLER_WORDS["en"])

    async def play_filler_words(self, duration: float = 1.0, interval: float = 0.5):
        """
        Play filler words during processing delays to maintain engagement
        """
        # Check if we should interrupt filler words
        if hasattr(self.session, '_current_speech') and self.session._current_speech is not None:
            return

        filler = random.choice(self.filler_words)
        # Use allow_interruptions=True for filler words so they can be stopped
        self.session.say(filler, add_to_chat_ctx=False, allow_interruptions=True)


    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        # Only play filler words if no current speech is active
        if not hasattr(self.session, '_current_speech') or self.session._current_speech is None:
            await self.play_filler_words()
