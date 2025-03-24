from datetime import datetime, timezone

from astropy import units
from astropy.coordinates import Angle


class Gen2:

    def __init__(self, actor=None, logger=None):
        self.actor = actor
        self.logger = logger
        self.timestamp = 0

    def receiveStatusKeys(self, key):
        self.timestamp = key.timestamp
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

    def _getValues(self, key):
        valueList = self.actor.models['gen2'].keyVarDict[key].valueList
        return {x.name: x.__class__.baseType(x) if x is not None else None for x in valueList} if len(
            valueList
            ) > 1 else valueList[0].__class__.baseType(valueList[0]) if valueList[0] is not None else None

    @property
    def visit(self):
        return self._getValues('visit')

    @property
    def inst_ids(self):
        return self._getValues('inst_ids')

    @property
    def program(self):
        return self._getValues('program')

    @property
    def object(self):
        return self._getValues('object')

    @property
    def pointing(self):
        return self._getValues('pointing')

    @property
    def offsets(self):
        return self._getValues('offsets')

    @property
    def conditions(self):
        return self._getValues('conditions')

    @property
    def coordinate_system_ids(self):
        return self._getValues('coordinate_system_ids')

    @property
    def tel_axes(self):
        return self._getValues('tel_axes')

    @property
    def tel_rot(self):
        return self._getValues('tel_rot')

    @property
    def tel_focus(self):
        return self._getValues('tel_focus')

    @property
    def tel_adc(self):
        return self._getValues('tel_adc')

    @property
    def dome_env(self):
        return self._getValues('dome_env')

    @property
    def outside_env(self):
        return self._getValues('outside_env')

    @property
    def m2(self):
        return self._getValues('m2')

    @property
    def m2rot(self):
        return self._getValues('m2rot')

    @property
    def pfuOffset(self):
        return self._getValues('pfuOffset')

    @property
    def frame_ids(self):
        return self._getValues('frame_ids')

    @property
    def autoguider(self):
        return self._getValues('autoguider')

    @property
    def header(self):
        return self._getValues('header')

    @property
    def statusUpdate(self):
        return self._getValues('statusUpdate')

    @property
    def tel_status(self):
        taken_at = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        tel_axes = self.tel_axes
        altitude = tel_axes['alt']
        azimuth = tel_axes['az']
        tel_rot = self.tel_rot
        insrot = tel_rot['instrot']
        inst_pa = tel_rot['posAngle']
        adc_pa = self.tel_adc['angle']
        m2_pos3 = self.m2['zpos']
        pointing = self.pointing
        cmd_ra = Angle(pointing['ra'], unit=units.hourangle).to(units.deg).value
        cmd_dec = Angle(pointing['dec'], unit=units.deg).value
        # dome_shutter_status
        # dome_light_status
        # created_at
        return altitude, azimuth, insrot, adc_pa, m2_pos3, cmd_ra, cmd_dec, inst_pa, taken_at

    @property
    def tel_dither(self):
        return self._getValues('telDither')

    @property
    def tel_guide(self):
        return self._getValues('telGuide')
