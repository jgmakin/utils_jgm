# standard libraries
import pdb
from functools import reduce

# third-party packages
import numpy as np
try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
except ModuleNotFoundError:
    print('WARNING: torch not found; proceeding without it...')

try:
    import tensorflow as tf
except ModuleNotFoundError:
    print('WARNING: tensorflow package not found; proceeding without it...')

# local
from utils_jgm.toolbox import auto_attribute, tau


class KinematicChains:
    @auto_attribute
    def __init__(
        self,
        motion_axes=[
            [0., 0., 1.],
            [0., 0., 1.],
            # [0, 0, 1]
        ],
        link_lengths=[12., 20.],
        joint_limits=[
            [-np.pi/2, np.pi/4],
            [np.pi/4,  3*np.pi/4],
            # [-np.pi/4, np.pi/4]
        ],
        end_effector_limits=None,
    ):

        # create all the joints
        self.joints = []
        for ind, (motion_axis, link_length, limits) in enumerate(
            zip(self.motion_axes, self.link_lengths, self.joint_limits)
        ):

            # set to single-entry list containing joint minimum
            self.joints.append(self.Joint(
                motion_axis=motion_axis,
                link_length=link_length,
                limits=limits,
                axis_location=[0., sum(self.link_lengths[:ind]), 0.],
                prev_joint=self.joints[-1] if self.joints else None,
                next_joint=None,
            ))

        ###
        # this is only necessary if you're actually using the linked list
        for ind, joint in enumerate(reversed(self.joints)):
            joint.next_joint = (
                self.joints[ind+1] if ind < len(self.joints)-1 else None
            )
        ###

        ###############
        # The following need to be assigned actual values in subclasses
        self.reference_config = None
        self._joint_angles = None
        ###############
        
    @property
    def position(self):
        # joint_angles wear the pants
        return self.forward_kinematics(self.joint_angles)

    @property
    def position_limits(self):
        raise NotImplementedError('shell property -- jgm')

    @position.setter
    def position(self, position):
        raise NotImplementedError('shell method -- jgm')
        # self._joint_angles = self.inverse_kinematics(position)
        # for joint, joint_angle in zip(self.joints, self._joint_angles):
        #     joint.displacement = joint_angle

    @property
    def joint_angles(self):
        return self._joint_angles

    @joint_angles.setter
    def joint_angles(self, joint_angles_matrix):
        self._joint_angles = joint_angles_matrix
        for joint, joint_angles in zip(self.joints, joint_angles_matrix.T):
            joint.angles = joint_angles

    def forward_kinematics(self, joint_angles):
        # the product of exponentials formulation
        rigid_body_configuration = reduce(
            lambda M1, M2: M2@M1, [
                joint.relative_screw(joint_angles[:, ind]) for
                # ind, joint in enumerate(reversed(self.joints))
                ind, joint in reversed(list(enumerate(self.joints)))
            ],
            self.reference_config
        )

        # just return the *position* of the end effector
        ####
        # NB: only return first two dimensions....
        p = rigid_body_configuration[:, :2, -1]
        ####

        return p

    def inverse_kinematics(self, position):
        raise NotImplementedError('shell method -- jgm')


class NumPyKinematicChains(KinematicChains):
    '''
    NB: Like the base class, KinematicChains, this class uses lists (and lists
    of lists), rather than numpy arrays, for numerical parameters. However, the
    joints (from the NumPyJoint class) that are attached to the object do use
    tensors.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        # maps points in base frame into tool frame when all joints are at 0
        # (any Joint's homogeneous_SE3 method will do)
        self.reference_config = self.joints[0].homogeneous_SE3(
            np.array([[[0], [sum(self.link_lengths)], [0]]]),
            np.eye(3)[None]
        )

        # make sure these are a numpy array
        self.joint_angles = np.array(self.joint_limits)[None, :, 0]

    @property
    def Joint(self):
        return NumPyJoint
    
    def forward_kinematics_Jacobian(self, joint_angles):
        '''
        fast Jacobian calculation
        '''

        J = np.zeros(
            joint_angles.shape[0],
            self.position.shape[1],
            self.joint_angles.shape[1],
        )

        L1 = self.link_lengths[0]
        L2 = self.link_lengths[1]
        total_angles = joint_angles.sum(axis=1)
        total_x = L2*np.cos(total_angles)
        total_y = L2*np.sin(total_angles)
        
        J[:, 0, 0] = -L1*np.cos(joint_angles[:, 0]) - total_x
        J[:, 0, 1] = -total_x
        J[:, 1, 0] = -L1*np.sin(joint_angles[:, 0]) - total_y
        J[:, 1, 1] = -total_y

    def inverse_kinematics(self, position):
        raise NotImplementedError('Oops! Inverse kinematics not written yet!!')


class TorchKinematicChains(KinematicChains):
    '''
    NB: Like the base class, KinematicChains, this class uses lists (and lists
    of lists), rather than tensors, for numerical parameters.  However, the
    joints (from the TorchJoint class) that are attached to the object do use
    tensors.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        # maps points in base frame into tool frame when all joints are at 0
        # (any Joint's homogeneous_SE3 method will do)
        self.reference_config = self.joints[0].homogeneous_SE3(
            torch.tensor([[[0], [sum(self.link_lengths)], [0]]]),
            torch.eye(3)[None]
        )

        # make sure these are a torch tensor
        self.joint_angles = torch.tensor(self.joint_limits)[None, :, 0]

    @property
    def Joint(self):
        return TorchJoint

    @property
    def position_limits(self):
        '''
        NB: This is something of a hack and won't work for all joint limits.
        But it will work for the default ones
        '''

        # leftmost, rightmost, bottommost
        left = self.forward_kinematics(torch.tensor(
            [[self.joint_limits[0][1], self.joint_limits[1][0]]]
        ))[0, 0].item()
        right = self.forward_kinematics(torch.tensor(
            [[self.joint_limits[0][0], self.joint_limits[1][0]]]
        ))[0, 0].item()
        bottom = self.forward_kinematics(torch.tensor(
            [[self.joint_limits[0][1], self.joint_limits[1][1]]]
        ))[0, 1].item()

        # topmost: compute longest reach by forming a right triangle
        La = self.link_lengths[1]*np.cos(tau/8) + self.link_lengths[0]
        Lb = self.link_lengths[1]*np.sin(tau/8)
        top = ((La**2 + Lb**2)**(1/2)).astype(np.float32)

        # [[min_x, min_y], [max_x, max_y]]
        return [[left, bottom], [right, top]]
        
    def forward_kinematics_Jacobian(self, joint_angles):
        '''
        fast Jacobian calculation
        '''

        J = torch.zeros(
            joint_angles.shape[0],
            self.position.shape[1],
            self.joint_angles.shape[1],
        )

        L1 = self.link_lengths[0]
        L2 = self.link_lengths[1]
        total_angles = joint_angles.sum(axis=1)
        total_x = L2*torch.cos(total_angles)
        total_y = L2*torch.sin(total_angles)
        
        J[:, 0, 0] = -L1*torch.cos(joint_angles[:, 0]) - total_x
        J[:, 0, 1] = -total_x
        J[:, 1, 0] = -L1*torch.sin(joint_angles[:, 0]) - total_y
        J[:, 1, 1] = -total_y

        return J

    def inverse_kinematics(self, positions):
        L1 = self.link_lengths[0]
        L2 = self.link_lengths[1]
        phi = torch.atan2(-positions[:, 0], positions[:, 1])
        rsq = (positions**2).sum(axis=1)
        gamma = (L1**2 + L2**2 - rsq)/(2*L1*L2)
        delta = (L1**2 - L2**2 + rsq)/(2*L1*rsq**(1/2))

        # cap these, lest you get imaginary numbers somewhere
        gamma = torch.clamp(gamma, -1, 1)
        delta = torch.clamp(delta, -1, 1)

        # ...
        alpha = torch.acos(gamma)
        beta = torch.acos(delta)
        joint_angles = torch.stack((phi - beta, torch.pi - alpha), axis=1)

        return joint_angles


# class TensorFlowKinematicChains(KinematicChains):

#     def __init__(
#         self,
#         motion_axes=[
#             [0, 0, 1],
#             [0, 0, 1],
#             # [0, 0, 1]
#         ],
#         link_lengths=[12, 20],
#         joint_limits=[
#             [-np.pi/2, np.pi/4],
#             [np.pi/4,  3*np.pi/4],
#             # [-np.pi/4, np.pi/4]
#         ],
#         end_effector_limits=None,
#     ):

#         # create all the joints
#         self.joints = []
#         for ind, (motion_axis, link_length, limits) in enumerate(
#             zip(motion_axes, link_lengths, joint_limits)
#         ):

#             # set to single-entry list containing joint minimum
#             these_angles = [limits[0]]
#             self.joints.append(TensorFlowJoint(
#                 angles=these_angles,
#                 motion_axis=motion_axis,
#                 link_length=link_length,
#                 limits=limits,
#                 axis_location=[0, sum(link_lengths[:ind]), 0],
#                 prev_joint=self.joints[-1] if self.joints else None,
#                 next_joint=None,
#             ))

#         ###
#         # this is only necessary if you're actually using the linked list
#         for ind, joint in enumerate(reversed(self.joints)):
#             joint.next_joint = (
#                 self.joints[ind+1] if ind < len(self.joints)-1 else None
#             )
#         ###

#         # maps points in base frame into tool frame when all joints are at 0
#         # [4 x 4] rather than [1 x 4 x 4] b/c tf doesn't broadcast matmul
#         self.reference_config = tf.concat(
#             (
#                 tf.concat((tf.eye(3), [[0.0], [sum(link_lengths)], [0.0]]), axis=1),
#                 [[0.0, 0.0, 0.0, 1.0]]
#             ), axis=0
#         )

#         # default--joint minima
#         self.joint_angles = tf.constant(
#             np.array(joint_limits)[None, :, 0], tf.float32
#         )

#     @property
#     def position(self):
#         return self._position

#     @property
#     def joint_angles(self):
#         return self._joint_angles

#     @position.setter
#     def position(self, position):
#         self._position = position
#         raise NotImplementedError('Oops!  Position setter doesn''t work yet')
#         # self._joint_angles = self.inverse_kinematics(position)
#         # for joint, joint_angle in zip(self.joints, self._joint_angles):
#         #     joint.displacement = joint_angle

#     @joint_angles.setter
#     def joint_angles(self, joint_angles_matrix):
#         self._joint_angles = joint_angles_matrix
#         for joint, joint_angles in zip(
#                 self.joints, tf.unstack(joint_angles_matrix, axis=1)):
#             joint.angles = joint_angles
#         self._position = self.forward_kinematics()

#     def forward_kinematics(self):
#         # the product of exponentials formulation
#         def batch_matmul(A, B):
#             if len(B.shape) == 2:
#                 return tf.tensordot(A, B, axes=1)
#             else:
#                 return A@B
    
#         rigid_body_configuration = reduce(
#             lambda M1, M2: batch_matmul(M2, M1),
#             [joint.relative_screw() for joint in reversed(self.joints)],
#             self.reference_config
#         )

#         # just return the *position* of the end effector
#         p = rigid_body_configuration[:, :3, -1]

#         return p

#     def inverse_kinematics(self, position):
#         raise NotImplementedError('Oops! Inverse kinematics not written yet!!')


class Joint:

    @auto_attribute
    def __init__(
        self,
        # angles=[-tau/4],
        motion_axis=[0., 0., 1.],
        link_length=12,
        limits=[-tau/4, tau/8],
        axis_location=[0., 0., 0.],
        prev_joint=None,
        next_joint=None,
    ):
        pass

    @property
    def motion_type(self):
        if any(self.motion_axis):
            return 'revolute'
        else:
            return 'prismatic'

    @property
    def axis_in_so3(self):
        w = self.motion_axis
        return [
            [    0, -w[2],  w[1]],
            [ w[2],     0, -w[0]],
            [-w[1],  w[0],     0]
        ]


class NumPyJoint(Joint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for attribute in ['motion_axis', 'axis_location']:
            setattr(self, attribute, np.array(getattr(self, attribute)))

    @property
    def axis_in_so3(self):
        return np.array(super().axis_in_so3)

    @property
    def unit_velocity(self):
        # force cross product to be a column vector
        return np.cross(self.axis_location, self.motion_axis)[:, None]

    def rodrigues(self, angles):
        theta = np.reshape(angles, [-1, 1, 1])
        omega_hat = self.axis_in_so3[None, :, :]

        return np.eye(3)[None] + omega_hat*np.sin(theta) + \
            omega_hat@omega_hat*(1 - np.cos(theta))

    def relative_screw(self, angles):
        R = self.rodrigues(angles)
        v = self.unit_velocity
        theta = np.reshape(angles, [-1, 1, 1])
        if self.motion_type == 'prismatic':
            p = v*theta
        else:  # revolute
            ###################
            # faster with einsum??
            ###################

            omega = np.reshape(self.motion_axis, [-1, 1])
            O = self.axis_in_so3

            # p = (np.eye(3) - R)@O@v + omega@omega.T@v*theta
            # compute p in a roundabout way to minimize matrix multiplications
            # p = (I - R)*W*v + w*w'*v*th
            #   = (I - (I + W*sin(th) + W^2*(1-cos(th))))*W*v + w*w'*v*th
            #   = (-W^2*v*sin(th) -W^3*v*(1-cos(th)) + w*w'*v*th
            Ov = O@v
            OOv = O@Ov
            OOOv = O@OOv
            ov = omega.T@v
            oov = omega*ov
            p = -OOv*np.sin(theta) - OOOv*(1-np.cos(theta)) + oov*theta

        return self.homogeneous_SE3(p, R)

    @staticmethod
    def homogeneous_SE3(p, R):
        '''
        From point p and rotation matrix R, which provide an affine
        transformation, construct a linear transformation (matrix), for rigid-
        body motion--the so-called homogeneous representation.

        See e.g. p. 36 of Murray/Li/Sastry.
        '''

        bottom_row = np.array([[0., 0., 0., 1.]])
        bottom_row = np.tile(bottom_row, [p.shape[0], 1, 1])
        return np.concatenate(
            (np.concatenate((R, p), axis=2), bottom_row), axis=1
        )


class TorchJoint(Joint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for attribute in ['motion_axis', 'axis_location']:
            setattr(self, attribute, torch.tensor(getattr(self, attribute)))

    @property
    def axis_in_so3(self):
        return torch.tensor(super().axis_in_so3)

    @property
    def unit_velocity(self):
        # force cross product to be a column vector
        return torch.cross(self.axis_location, self.motion_axis, dim=0)[:, None]

    def rodrigues(self, angles):
        theta = torch.reshape(angles, [-1, 1, 1])
        omega_hat = self.axis_in_so3[None, :, :]

        return torch.eye(3)[None] + omega_hat*torch.sin(theta) + \
            omega_hat@omega_hat*(1 - torch.cos(theta))

    def relative_screw(self, angles):
        R = self.rodrigues(angles)
        v = self.unit_velocity
        theta = torch.reshape(angles, [-1, 1, 1])
        if self.motion_type == 'prismatic':
            p = v*theta
        else:  # revolute
            omega = torch.reshape(self.motion_axis, [-1, 1])
            O = self.axis_in_so3

            # p = (np.eye(3) - R)@O@v + omega@omega.T@v*theta
            # compute p in a roundabout way to minimize matrix multiplications
            # p = (I - R)*W*v + w*w'*v*th
            #   = (I - (I + W*sin(th) + W^2*(1-cos(th))))*W*v + w*w'*v*th
            #   = (-W^2*v*sin(th) -W^3*v*(1-cos(th)) + w*w'*v*th
            Ov = O@v
            OOv = O@Ov
            OOOv = O@OOv
            ov = omega.T@v
            oov = omega*ov
            p = -OOv*torch.sin(theta) - OOOv*(1-torch.cos(theta)) + oov*theta

        return self.homogeneous_SE3(p, R)

    @staticmethod
    def homogeneous_SE3(p, R):
        '''
        From point p and rotation matrix R, which provide an affine
        transformation, construct a linear transformation (matrix), for rigid-
        body motion--the so-called homogeneous representation.

        See e.g. p. 36 of Murray/Li/Sastry.
        '''

        bottom_row = torch.tensor([[0, 0, 0, 1]])
        bottom_row = torch.tile(bottom_row, [p.shape[0], 1, 1])
        return torch.concatenate(
            (torch.concatenate((R, p), axis=2), bottom_row), axis=1
        )


class TensorFlowJoint(Joint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for attribute in ['motion_axis', 'axis_location']:
            setattr(
                self, attribute,
                tf.constant(getattr(self, attribute), tf.float32)
            )

    @property
    def motion_type(self):
        if any(self.motion_axis):
            return 'revolute'
        else:
            return 'prismatic'

    @property
    def axis_in_so3(self):
        ### will this really work??
        return np.array(super().axis_in_so3)

    @property
    def unit_velocity(self):
        velocity = tf.cross(
            tf.constant(self.axis_location),
            tf.constant(self.motion_axis)
        )
        return tf.expand_dims(tf.cast(velocity, tf.float32), axis=1)

    def rodrigues(self, angles):
        theta = tf.expand_dims(tf.expand_dims(angles, 1), 2)
        omega_hat = tf.reshape(self.axis_in_so3, [1, 3, 3])

        return tf.eye(3, batch_shape=[1]) + omega_hat*tf.sin(theta) + \
            omega_hat@omega_hat*(1 - tf.cos(theta))

    def relative_screw(self, angles):
        R = self.rodrigues(angles)
        v = self.unit_velocity
        theta = tf.expand_dims(tf.expand_dims(angles, 1), 2)
        if self.motion_type == 'prismatic':
            p = v*theta
        else:  # revolute
            O = self.axis_in_so3
            omega = tf.constant(np.array([self.motion_axis]).T, tf.float32)

            # p = (np.eye(3) - R)@O@v + omega@omega.T@v*theta
            # compute p in a roundabout way to minimize matrix multiplications
            # p = (I - R)*W*v + w*w'*v*th
            #   = (I - (I + W*sin(th) + W^2*(1-cos(th))))*W*v + w*w'*v*th
            #   = (-W^2*v*sin(th) -W^3*v*(1-cos(th)) + w*w'*v*th
            Ov = O@v
            OOv = O@Ov
            OOOv = O@OOv
            ov = tf.transpose(omega)@v
            oov = omega*ov
            p = -OOv*tf.sin(theta) - OOOv*(1-tf.cos(theta)) + oov*theta

        return self.homogeneous_SE3(p, R)

    @staticmethod
    def homogeneous_SE3(p, R):
        '''
        From point p and rotation matrix R, which provide an affine
        transformation, construct a linear transformation (matrix), for rigid-
        body motion--the so-called homegeneous representation.

        See e.g. p. 36 of Murray/Li/Sastry.
        '''

        bottom_row = tf.constant([[[0, 0, 0, 1.0]]])
        bottom_row = tf.tile(bottom_row, (p.shape[0], 1, 1))
        return tf.concat((tf.concat((R, p), axis=2), bottom_row), axis=1)