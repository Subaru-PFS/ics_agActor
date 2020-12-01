class Mlp1:

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

        valueList = self.actor.models['mlp1'].keyVarDict[key].valueList
        return {x.name: x.__class__.baseType(x) for x in valueList} if len(valueList) > 1 else valueList[0].__class__.baseType(valueList[0])

    @property
    def telescopeState(self):

        return self._getValues('telescopeState')
