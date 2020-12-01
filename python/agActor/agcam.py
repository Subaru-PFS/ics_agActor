class Agcam:

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

        valueList = self.actor.models['agcam'].keyVarDict[key].valueList
        return {x.name: x.__class__.baseType(x) for x in valueList} if len(valueList) > 1 else valueList[0].__class__.baseType(valueList[0])

    @property
    def exposureState(self):

        return self._getValues('exposureState')

    @property
    def exposureTime(self):

        return self._getValues('exposureTime')

    @property
    def frameId(self):

        return self._getValues('frameId')

    @property
    def filepath(self):

        return self._getValues('filepath')

    @property
    def cameraState1(self):

        return self._getValues('cameraState1')

    @property
    def cameraState2(self):

        return self._getValues('cameraState2')

    @property
    def cameraState3(self):

        return self._getValues('cameraState3')

    @property
    def cameraState4(self):

        return self._getValues('cameraState4')

    @property
    def cameraState5(self):

        return self._getValues('cameraState5')

    @property
    def cameraState6(self):

        return self._getValues('cameraState6')

    @property
    def dataTime(self):

        return self._getValues('dataTime')

    @property
    def detectionState(self):

        return self._getValues('detectionState')
