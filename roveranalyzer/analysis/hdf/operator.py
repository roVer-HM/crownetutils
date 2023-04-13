class Operation:
    GREATER = ">"
    SMALLER = "<"
    GREATER_EQ = ">="
    SMALLER_EQ = "<="
    EQ = "="

    @classmethod
    def Equal(cls, value):
        return cls(cls.EQ, value)

    @classmethod
    def GT(cls, value):
        return cls(cls.GREATER, value)

    @classmethod
    def GTE(cls, value):
        return cls(cls.EQ, value)

    @classmethod
    def LT(cls, value):
        return cls(cls.SMALLER, value)

    @classmethod
    def LTE(cls, value):
        return cls(cls.SMALLER_EQ, value)

    def __init__(self, operator, value):
        self._op = operator
        self._value = value

    @property
    def operation(self):
        return self._op

    @property
    def value(self):
        return self._value
