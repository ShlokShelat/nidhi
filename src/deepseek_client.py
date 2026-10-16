#!/usr/bin/env python3
"""
deepseek_client.py

Independent verifier client. Mirrors openai_client.py's public
interface (generate, generate_json) so verify_generated.py can use
either interchangeably — but this one talks to DeepSeek's API, a
genuinely different model family from GPT-5, for real cross-model
verification instead of GPT-5 grading its own generations.

DeepSeek's API is OpenAI-SDK-compatible via base_url override, but
uses the standard Chat Completions endpoint (messages=[...]), not
OpenAI's newer Responses API (instructions=/input=) that
openai_client.py uses. Hence a separate client rather than a config
flag on OpenAIClient.

Model name: defaults to "deepseek-chat". If your account uses a
different name (e.g. "deepseek-reasoner"), override via
GenerationConfig(model=...) or DEEPSEEK_MODEL env var.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time

from dataclasses import dataclass
from typing import Any, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)

###############################################################################
# Logging
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

###############################################################################
# Defaults
###############################################################################

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

DEFAULT_TEMPERATURE = 0.0  # verification should be as deterministic as possible

DEFAULT_MAX_OUTPUT_TOKENS = 1024

DEFAULT_TIMEOUT = 120

DEFAULT_MAX_RETRIES = 5

###############################################################################
# Configuration
###############################################################################

@dataclass(slots=True)
class GenerationConfig:

    model: str = DEFAULT_MODEL

    temperature: Optional[float] = DEFAULT_TEMPERATURE

    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    timeout: int = DEFAULT_TIMEOUT

    max_retries: int = DEFAULT_MAX_RETRIES

###############################################################################
# DeepSeek Client
###############################################################################

class DeepSeekClient:

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
    ):

        self.config = config or GenerationConfig()

        api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:

            raise EnvironmentError(
                "\nDEEPSEEK_API_KEY not found.\n"
                "Run:\n"
                "export DEEPSEEK_API_KEY='sk-...'\n"
            )

        self.client = OpenAI(

            api_key=api_key,

            base_url=DEEPSEEK_BASE_URL,

            timeout=self.config.timeout,

        )

        logger.info(
            "Initialized DeepSeek client (%s) via %s",
            self.config.model,
            DEEPSEEK_BASE_URL,
        )

    ###########################################################################
    # Retry
    ###########################################################################

    def _retry_sleep(self, attempt: int):

        delay = (2 ** attempt) + random.random()

        logger.warning(
            "Retrying in %.2f seconds...",
            delay,
        )

        time.sleep(delay)

    ###########################################################################
    # Core Request (Chat Completions shape, not Responses API)
    ###########################################################################

    def _request(

        self,

        system_prompt: str,

        user_prompt: str,

    ):

        kwargs = {

            "model": self.config.model,

            "messages": [

                {"role": "system", "content": system_prompt},

                {"role": "user", "content": user_prompt},

            ],

            "max_tokens": self.config.max_output_tokens,

        }

        if self.config.temperature is not None:

            kwargs["temperature"] = self.config.temperature

        last_exception = None

        for attempt in range(self.config.max_retries):

            try:

                return self.client.chat.completions.create(**kwargs)

            except (

                APIConnectionError,

                APIStatusError,

                RateLimitError,

            ) as e:

                last_exception = e

                logger.warning(

                    "DeepSeek request failed (%d/%d): %s",

                    attempt + 1,

                    self.config.max_retries,

                    e,

                )

                if attempt != self.config.max_retries - 1:

                    self._retry_sleep(attempt)

        raise RuntimeError(
            "Maximum retries exceeded."
        ) from last_exception

    ###########################################################################
    # Public API — same shape as OpenAIClient
    ###########################################################################

    def generate(

        self,

        system_prompt: str,

        user_prompt: str,

    ) -> str:

        response = self._request(

            system_prompt,

            user_prompt,

        )

        return response.choices[0].message.content.strip()

    ###########################################################################
    # JSON Helpers — same robustness logic as openai_client.py
    ###########################################################################

    @staticmethod
    def _extract_json(text: str) -> Any:

        text = text.strip()

        if text.startswith("```"):

            text = re.sub(
                r"^```(?:json)?",
                "",
                text,
                flags=re.IGNORECASE,
            )

            text = re.sub(
                r"```$",
                "",
                text,
            )

            text = text.strip()

        try:

            parsed = json.loads(text)

        except json.JSONDecodeError:

            print("=" * 80)
            print("RAW MODEL OUTPUT (DeepSeek)")
            print("=" * 80)
            print(text)
            print("=" * 80)

            raise

        if isinstance(parsed, dict):

            if "posts" in parsed:

                return parsed["posts"]

            if "data" in parsed:

                return parsed["data"]

        return parsed

    ###########################################################################

    def generate_json(

        self,

        system_prompt: str,

        user_prompt: str,

    ):

        text = self.generate(

            system_prompt,

            user_prompt,

        )

        return self._extract_json(text)

    ###########################################################################
    # Configuration Printer
    ###########################################################################

    def print_configuration(self):

        print("=" * 80)

        print("DeepSeek Client Configuration")

        print("=" * 80)

        print()

        print(f"Model               : {self.config.model}")

        print(f"Base URL            : {DEEPSEEK_BASE_URL}")

        print(f"Temperature         : {self.config.temperature}")

        print(f"Max Output Tokens   : {self.config.max_output_tokens}")

        print(f"Timeout             : {self.config.timeout}")

        print(f"Max Retries         : {self.config.max_retries}")

        print()

###############################################################################
# Demo
###############################################################################

def main():

    client = DeepSeekClient()

    client.print_configuration()

    system_prompt = "You are a helpful assistant."

    user_prompt = (
        "Return valid JSON.\n\n"
        '{'
        '"posts":["Example 1","Example 2"]'
        '}'
    )

    print("=" * 80)

    print("Testing JSON Generation")

    print("=" * 80)

    result = client.generate_json(

        system_prompt,

        user_prompt,

    )

    print(result)

    print()

    print("=" * 80)

    print("Success")

    print("=" * 80)

if __name__ == "__main__":

    main()
