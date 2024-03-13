import utils.functions as f


class Tools:
    @classmethod
    def timeit(cls, func):
        return cls.callback(func, "timeit")

    @classmethod
    def retry(cls, func):
        return cls.callback(func, "retry")

    @classmethod
    def cache(cls, func):
        return cls.callback(func, "cache")

    @classmethod
    def log(cls, func):
        return cls.callback(func, "log")

    @classmethod
    def single(cls, _cls):
        return cls.callback(_cls, "single")

    @classmethod
    def valid_type(cls, func):
        return cls.callback(func, "valid_type")
    @classmethod
    def callback(cls, func, type):
        match type:
            case "timeit": return f.__timer__(func)
            case "retry": return f.__retry__(func)
            case "lru_cache": return f.__cache__(func)
            case "log": return f.__log__(func)
            case "single": return f.__single__(func)
            case "valid_type": return f.__valid_type__(func)