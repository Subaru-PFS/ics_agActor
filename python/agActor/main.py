#!/usr/bin/env python

import argparse
import queue
from actorcore.ICC import ICC
from agActor.agcc import Agcc
from agActor.mlp1 import Mlp1


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

            self.agcc = Agcc(actor=self, logger=self.logger)
            self.mlp1 = Mlp1(actor=self, logger=self.logger)

            _models = ('agcc', 'mlp1',)
            self.addModels(_models)
            #self.models['agcc'].keyVarDict[''].addCallback(self.agcc.receiveStatusKeys, callNow=False)
            #self.models['mlp1'].keyVarDict[''].addCallback(self.mlp1.receiveStatusKeys, callNow=False)

    # override
    def connectionLost(self, reason):

        pass

    # override
    def commandFailed(self, cmd):

        pass

    def sendCommand(self, actor=None, cmdStr=None, timeLim=0, callFunc=None, **kwargs):

        if callFunc is None:
            self.logger.info('calling Cmdr.cmdq...')
            q = self.cmdr.cmdq(actor=actor, cmdStr=cmdStr, timeLim=timeLim, **kwargs)
            while True:
                try:
                    result = q.get(timeout=1)
                    break
                except queue.Empty:
                    if not self.cmdr.connector.activeConnection:
                        raise Exception('connection lost: actor={},cmdStr="{}",timeLim={},kwargs={}'.format(actor, cmdStr, timeLim, str(kwargs)))
            for reply in result.replyList:
                self.logger.info('reply={}'.format(reply.canonical()))
            self.logger.info('didFail={}'.format(result.didFail))
            if result.didFail:
                raise Exception('command failed: actor={},cmdStr="{}",timeLim={},kwargs={}'.format(actor, cmdStr, timeLim, str(kwargs)))
            return result
        else:
            self.logger.info('calling Cmdr.bgCall...')
            self.cmdr.bgCall(callFunc=callFunc, actor=actor, cmdStr=cmdStr, timeLim=timeLim, **kwargs)
            return None


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
