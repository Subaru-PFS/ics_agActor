#!/usr/bin/env python

import argparse
import queue
from actorcore.ICC import ICC
from agActor.models.agcc import Agcc
from agActor.models.mlp1 import Mlp1
from agActor.models.gen2 import Gen2


class AgActor(ICC):

    # Keyword arguments for this class
    _kwargs = {
    }

    def __init__(self, name, **kwargs):

        # Consume keyword arguments for this class
        for k in AgActor._kwargs:
            if k in kwargs:
                setattr(self, '_' + k, kwargs[k])
                del kwargs[k]
            else:
                setattr(self, '_' + k, AgActor._kwargs[k])

        super().__init__(name, **kwargs)

        self._everConnected = False

    # override
    def shutdown(self):

        self.stopAllControllers()
        super()._shutdown()

    def reloadConfiguration(self, cmd):

        pass

    # override
    def connectionMade(self):

        if not self._everConnected:

            self._everConnected = True

            #self.allControllers = ['ac', 'af', 'ag',]
            self.allControllers = ['ag',]
            self.attachAllControllers()

            self.agcc = Agcc('agcc', actor=self, logger=self.logger)
            self.mlp1 = Mlp1('mlp1', actor=self, logger=self.logger)
            self.gen2 = Gen2('gen2', actor=self, logger=self.logger)

            _models = ('agcc', 'mlp1', 'gen2',)
            self.addModels(_models)
            #self.models['agcc'].keyVarDict[''].addCallback(self.agcc.receiveStatusKeys, callNow=False)
            #self.models['mlp1'].keyVarDict[''].addCallback(self.mlp1.receiveStatusKeys, callNow=False)
            self.models['gen2'].keyVarDict['tel_axes'].addCallback(self.gen2.receiveStatusKeys, callNow=False)  # for timestamp only

    # override
    def connectionLost(self, reason):

        pass

    # override
    def commandFailed(self, cmd):

        pass

    def queueCommand(self, actor=None, cmdStr=None, timeLim=0, **kwargs):

        params = {k: v for k, v in locals().items() if k not in ('self',)}
        result = self.cmdr.cmdq(actor=actor, cmdStr=cmdStr, timeLim=timeLim, **kwargs)

        class _Result:

            def __init__(self, cmdr, logger):

                self.connector = cmdr.connector
                self.logger = logger
                self.params = params
                self.result = result

            def __del__(self):

                #self.logger.info('_Result.__del__:')
                del self.connector

            def get(self):

                while True:
                    try:
                        result = self.result.get(timeout=0.1)
                        break
                    except queue.Empty:
                        if not self.connector.isConnected():
                            raise Exception('connection lost: params={}'.format(self.params))
                for reply in result.replyList:
                    self.logger.info('reply={}'.format(reply.canonical()))
                self.logger.info('didFail={}'.format(result.didFail))
                if result.didFail:
                    raise Exception('command failed: params={}'.format(self.params))
                return result

        return _Result(self.cmdr, self.logger)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--configFile', default=None)
    args = parser.parse_args()

    actor = AgActor(
        'ag',
        productName='agActor',
        configFile=args.configFile
    )
    try:
        actor.run()
    except:
        raise
    finally:
        actor.shutdown()


if __name__ == '__main__':

    main()
