import numpy as np
import scipy
import sys
import os
import glob
import torch

from scipy.misc import imresize

import matplotlib.pyplot as plt



from network import CoILModel

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from challenge.autonomous_agent import AutonomousAgent


class CoILAgent(AutonomousAgent):

    def __init__(self):
        AutonomousAgent.__init__(self)
        params = {
            'checkpoint': '320000',
            "model_type": 'coil-icra',
            "model_configuration": {'perception': {
                'res': {
                    'name': 'resnet34',
                    'num_classes': 512
                }
            },
                'measurements': {
                    'fc': {
                        'neurons': [128, 128],
                        'dropouts': [0.0, 0.0]
                    }
                },
                'join': {
                    'fc': {
                        'neurons': [512],
                        'dropouts': [0.0]
                    }
                },
                'speed_branch': {
                    'fc': {
                        'neurons': [256, 256],
                        'dropouts': [0.0, 0.5]
                    }
                },
                'branches': {
                    'number_of_branches': 4,
                    'fc': {
                        'neurons': [256, 256],
                        'dropouts': [0.0, 0.5]
                    }
                }
            },
            'image_cut': [90, 485],
            'speed_factor': 12.0,
            'size': [3, 88, 200]
        }

        self._checkpoint = torch.load(str(params['checkpoint']) + '.pth')
        # Set the carla version that is going to be used by the interface
        # We save the checkpoint for some interesting future use.

        self._model = CoILModel(params["model_type"],
                                params["model_configuration"])
        self.first_iter = True
        # Load the model and prepare set it for evaluation
        self._model.load_state_dict(self._checkpoint['state_dict'])
        self._model.cuda()
        self._model.eval()

        self.latest_image = None
        self.latest_image_tensor = None


    def sensors_setup(self):
        sensors = [['sensor.camera.rgb',
                   {'x': 2.0, 'y': 0.0,
                    'z': 1.40, 'roll': 0.0,
                    'pitch': -15.0, 'yaw': 0.0,
                    'width': 800, 'height': 600,
                    'fov': 100},
                    'rgb'],
                   ['sensor.speedometer',
                    {'reading_frequency': 20},
                    'speed'
                    ]
                   ]

        return sensors

    def run_step(self, input_data):

        # TODO, input speed should come on the input data

        #measurements, sensor_data, directions, target
        print ("Input data")
        print (input_data)

        directions = 2.0  # function to get directions from the plan

        # Take the forward speed and normalize it for it to go from 0-1
        norm_speed = input_data['speed'] / self._params['speed_factor']#.SPEED_FACTOR
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
        directions_tensor = torch.cuda.LongTensor([directions])
        # Compute the forward pass processing the sensors got from CARLA.
        model_outputs = self._model.forward_branch(self._process_sensors(input_data['rgb']), norm_speed,
                                                   directions_tensor)

        steer, throttle, brake = self._process_model_outputs(model_outputs[0])

        control = carla.VehicleControl()

        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        # There is the posibility to replace some of the predictions with oracle predictions.

        self.first_iter = False

        return control

    def get_attentions(self, layers=None):
        """

        Returns
            The activations obtained from the first layers of the latest iteration.

        """
        if layers is None:
            layers = [0, 1, 2]
        if self.latest_image_tensor is None:
            raise ValueError('No step was ran yet. '
                             'No image to compute the activations, Try Running ')
        all_layers = self._model.get_perception_layers(self.latest_image_tensor)
        cmap = plt.get_cmap('inferno')
        attentions = []
        for layer in layers:
            y = all_layers[layer]
            att = torch.abs(y).mean(1)[0].data.cpu().numpy()
            att = att / att.max()
            att = cmap(att)
            att = np.delete(att, 3, 2)
            attentions.append(imresize(att, [88, 200]))
        return attentions

    def _process_sensors(self, sensor):

        iteration = 0
        # TODO check the sensor itens things

        sensor = sensor[self._params['image_cut'][0]:self._params['image_cut'][1], ...]

        sensor = scipy.misc.imresize(sensor, (self._params['size'][1], self._params['size'][2]))

        self.latest_image = sensor

        sensor = np.swapaxes(sensor, 0, 1)

        sensor = np.transpose(sensor, (2, 1, 0))

        sensor = torch.from_numpy(sensor / 255.0).type(torch.FloatTensor).cuda()

        if iteration == 0:
            image_input = sensor

        iteration += 1

        image_input = image_input.unsqueeze(0)

        self.latest_image_tensor = image_input

        return image_input

    def _process_model_outputs(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        steer, throttle, brake = outputs[0], outputs[1], outputs[2]
        if brake < 0.05:
            brake = 0.0

        if throttle > brake:
            brake = 0.0


        return steer, throttle, brake


    def _process_model_outputs_wp(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        wpa1, wpa2, throttle, brake = outputs[3], outputs[4], outputs[1], outputs[2]
        if brake < 0.2:
            brake = 0.0

        if throttle > brake:
            brake = 0.0

        steer = 0.7 * wpa2

        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        return steer, throttle, brake

    def _get_oracle_prediction(self, measurements, target):
        # For the oracle, the current version of sensor data is not really relevant.
        control, _, _, _, _ = self.control_agent.run_step(measurements, [], [], target)

        return control.steer, control.throttle, control.brake