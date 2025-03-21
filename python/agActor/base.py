class BaseModel:

    def __init__(self, model_name, actor=None, logger=None):

        self.model_name = model_name
        self.actor = actor
        self.logger = logger

    def receiveStatusKeys(self, key):

        self.logger.info(
            'receiveStatusKeys: {},{},{},{},{},{}'.format(
                key.actor,
                key.name,
                key.timestamp,
                key.isCurrent,
                key.isGenuine,
                [x.__class__.baseType(x) if x is not None else None for x in key.valueList]
            )
        )

        if all((key.name == '', key.isCurrent, key.isGenuine)):
            pass

    def _getValues(self, key):

        valueList = self.actor.models[self.model_name].keyVarDict[key].valueList

        if len(valueList) == 0:
            return None
        if len(valueList) == 1 and valueList[0] is not None:
            return valueList[0].__class__.baseType(valueList[0])
        if len(valueList) > 1:
            return {x.name: x.__class__.baseType(x) if x is not None else None for x in valueList}
        return None
