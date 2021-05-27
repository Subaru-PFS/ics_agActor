class Agcc:

    def __init__(self, actor=None, logger=None):

        self.actor = actor
        self.logger = logger

    def receiveStatusKeys(self, key):

        self.logger.info('receiveStatusKeys: {},{},{},{},{},{}'.format(
            key.actor,
            key.name,
            key.timestamp,
            key.isCurrent,
            key.isGenuine,
            [x.__class__.baseType(x) for x in key.valueList]
        ))

        if all((key.name == '', key.isCurrent, key.isGenuine)):
            pass

    def _getValues(self, key):

        valueList = self.actor.models['agcc'].keyVarDict[key].valueList
        return {x.name: x.__class__.baseType(x) for x in valueList} if len(valueList) > 1 else valueList[0].__class__.baseType(valueList[0])

    @property
    def filepath(self):

        return self._getValues('agc_fitsfile')['filename']

    @property
    def frameId(self):

        #return self._getValues('agc_frameid')
        import random
        return random.randint(1, 9999)

    @property
    def dataTime(self):

        return self._getValues('agc_fitsfile')['timestamp']
