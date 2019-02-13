
import unittest
import random
import time
from ..challenge.data_provider import CallBack, Speedometer

import carla


# Fake vehicle class for testing purposes

class FakeVehicle():

    def get_transform(self):
        x = random.randint(0,10000)/10
        y = random.randint(0,10000)/10
        z = random.randint(0,10000)/10

        roll = random.randint(0, 100)
        pitch = random.randint(0, 100)
        yaw = random.randint(0, 100)


        return carla.Transform(location=carla.Location(x,y,z), rotation=carla.Rotation(roll,pitch,yaw))

    def get_velocity(self):

        x = random.randint(0,40)
        y = random.randint(0,40)
        z = random.randint(0,40)

        return carla.Vector3D(x,y,z)


def call_back_function(data):
    print (data)

class testSpeedometer(unittest.TestCase):
    def test_default_values(self):

        sensor = Speedometer(FakeVehicle(), 1)

        sensor.listen(call_back_function)




        time.spleep(10)



