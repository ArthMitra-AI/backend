from functools import lru_cache

class Settings:
    groq_api_key: str = ""
    gemini_api_key: str = ""
    gemini_flash_model: str = "gemini-2.5-flash"
    gemini_pro_model: str = "gemini-2.5-flash"
    openai_api_key: str = ""
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_key: str = ""
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    environment: str = "development"
    portfolio_scan_per_hour: int = 5

    def __init__(self):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        for key in vars(self.__class__):
            if not key.startswith('_'):
                env_val = os.getenv(key.upper())
                if env_val:
                    setattr(self, key, env_val)

@lru_cache()
def get_settings():
    return Settings()