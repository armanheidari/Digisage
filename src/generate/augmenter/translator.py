from googletrans import Translator


class Translator:
    _translator = Translator()

    @classmethod
    async def translate(cls, text: str, src: str, dest: str) -> str:
        translation = await cls._translator.translate(text, src=src, dest=dest)
        return translation.text