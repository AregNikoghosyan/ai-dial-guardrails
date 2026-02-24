import re
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from pydantic import SecretStr

from tasks._constants import DIAL_URL, API_KEY

class PresidioStreamingPIIGuardrail:

    def __init__(self, buffer_size: int = 100, safety_margin: int = 20):

        # 1. Language configuration
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_sm"}
            ],
        }

        provider = NlpEngineProvider(nlp_configuration=configuration)

        self.analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=["en"]
        )

        self.anonymizer = AnonymizerEngine()

        self.buffer = ""

        self.buffer_size = buffer_size
        self.safety_margin = safety_margin

    def process_chunk(self, chunk: str) -> str:

        if not chunk:
            return chunk

        self.buffer += chunk

        if len(self.buffer) > self.buffer_size:

            safe_length = len(self.buffer) - self.safety_margin

            for i in range(safe_length - 1, max(0, safe_length - 20), -1):
                if self.buffer[i] in ' \n\t.,;:!?':
                    safe_length = i
                    break

            text_to_process = self.buffer[:safe_length]

            results = self.analyzer.analyze(
                text=text_to_process,
                language="en"
            )

            anonymized = self.anonymizer.anonymize(
                text=text_to_process,
                analyzer_results=results
            )

            self.buffer = self.buffer[safe_length:]

            return anonymized.text

        return ""

    def finalize(self) -> str:

        if not self.buffer:
            return ""

        results = self.analyzer.analyze(
            text=self.buffer,
            language="en"
        )

        anonymized = self.anonymizer.anonymize(
            text=self.buffer,
            analyzer_results=results
        )

        self.buffer = ""

        return anonymized.text

class StreamingPIIGuardrail:

    def __init__(self, buffer_size: int = 100, safety_margin: int = 20):
        self.buffer_size = buffer_size
        self.safety_margin = safety_margin
        self.buffer = ""

    @property
    def _pii_patterns(self):
        return {
    'ssn': (
        r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        '[REDACTED-SSN]'
    ),
    'credit_card': (
        r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b',
        '[REDACTED-CREDIT-CARD]'
    ),
    'license': (
        r'\b[A-Z]{2}-DL-[A-Z0-9]+\b',
        '[REDACTED-LICENSE]'
    ),
    'bank_account': (
        r'\b(?:Bank\s+of\s+\w+\s*[-\s]*)?(?<!\d)\d{10,12}(?!\d)\b',
        '[REDACTED-ACCOUNT]'
    ),
    'iso_date': (
        r'\b\d{4}-\d{2}-\d{2}\b',
        '[REDACTED-DATE]'
    ),
    'long_date': (
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        '[REDACTED-DATE]'
    ),
    'expiry': (
        r'\b\d{2}/\d{2}\b',
        '[REDACTED-EXPIRY]'
    ),
    'cvv': (
        r'\b\d{3,4}\b',
        '[REDACTED-CVV]'
    ),
    'address': (
        r'\b\d+\s+[A-Za-z0-9\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}\b',
        '[REDACTED-ADDRESS]'
    ),
    'currency': (
        r'\$[\d,]+\.?\d*',
        '[REDACTED-AMOUNT]'
    ),
    'plain_large_number': (
        r'\b\d{5,}\b',
        '[REDACTED-NUMBER]'
    )
}
    def _detect_and_redact_pii(self, text: str) -> str:
        cleaned_text = text
        for pattern, replacement in self._pii_patterns.values():
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        return cleaned_text

    def process_chunk(self, chunk: str) -> str:
        if not chunk:
            return chunk

        self.buffer += chunk

        if len(self.buffer) > self.buffer_size:
            safe_length = len(self.buffer) - self.safety_margin
            text_to_output = self.buffer[:safe_length]
            safe_output = self._detect_and_redact_pii(text_to_output)
            self.buffer = self.buffer[safe_length:]
            return safe_output

        return ""

    def finalize(self) -> str:
        if self.buffer:
            final_output = self._detect_and_redact_pii(self.buffer)
            self.buffer = ""
            return final_output
        return ""


SYSTEM_PROMPT = "You are a helpful assistant."

PROFILE = """
# Profile: Amanda Grace Johnson

**Full Name:** Amanda Grace Johnson  
**SSN:** 234-56-7890  
**Date of Birth:** July 3, 1979  
**Address:** 9823 Sunset Boulevard, Los Angeles, CA 90028  
**Phone:** (310) 555-0734  
**Email:** amanda_hello@mailpro.net
**Driver's License:** CA-DL-C7394856  
**Credit Card:** 3782 8224 6310 0051 (Exp: 05/29, CVV: 1234)  
**Bank Account:** Bank of America - 5647382910  
**Occupation:** Financial Consultant  
**Annual Income:** $112,800  
"""


def main():

    llm = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="2024-02-15-preview",
        model="gpt-4.1-nano-2025-04-14",
        temperature=0,
        streaming=True,
    )

    guardrail = StreamingPIIGuardrail()

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=PROFILE),
    ]

    print("Streaming PII Guardrail Active.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        messages.append(HumanMessage(content=user_input))

        print("\nAssistant: ", end="", flush=True)

        full_response = ""

        for chunk in llm.stream(messages):
            if chunk.content:
                safe_chunk = guardrail.process_chunk(chunk.content)
                if safe_chunk:
                    print(safe_chunk, end="", flush=True)
                    full_response += safe_chunk

        remaining = guardrail.finalize()
        if remaining:
            print(remaining, end="", flush=True)
            full_response += remaining

        print("\n")

        messages.append(AIMessage(content=full_response))


if __name__ == "__main__":
    main()