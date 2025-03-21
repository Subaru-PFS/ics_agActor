from agActor.base import BaseModel


class Mlp1(BaseModel):

    @property
    def telescopeState(self):

        return self._getValues('telescopeState')

    @staticmethod
    def setUnixDay(t, t0):

        t %= 86400
        t += t0 // 86400 * 86400
        if 43200 <= t - t0:
            t -= 86400
        elif t - t0 < -43200:
            t += 86400
        return t
