"""
SOFA-based physics simulation that generates the dense per-vertex contact
force supervision used to train EgoPHI on ARCTIC (see the paper's force
simulation pipeline). Requires SOFA (https://www.sofa-framework.org/) with
the SofaPython3 plugin.

For each ARCTIC sequence, drops the recorded hand/object mesh trajectories
into a physics scene and lets SOFA's collision response compute per-vertex
contact forces frame by frame, saving them under config.FORCE_ROOT in the
layout arctic_preprocess.py / dataset_ARCTIC.py expect.
"""

import os

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import trimesh

import config

SofaRuntime.importPlugin("SofaComponentAll")
SofaRuntime.importPlugin("SofaPython3")

# The per-sequence single-mesh .obj files (first-frame hand/object meshes
# used to build the simulated bodies) live alongside PROCESSED_SEQS_ROOT's
# per-sequence trajectory .npy files, under a sibling "first_frame" folder.
FIRST_FRAME_ROOT = os.environ.get(
    "EGOPHI_FORCE_SIM_FIRST_FRAME_ROOT",
    os.path.join(os.path.dirname(config.PROCESSED_SEQS_ROOT), "first_frame"),
)

OBJECT_MASS = {
    'laptop': 0.8, 'phone': 0.1, 'ketchup': 0.05, 'mixer': 0.85,
    'waffleiron': 0.8, 'scissors': 0.09, 'capsulemachine': 0.4, 'box': 0.75,
    'notebook': 0.5, 'espressomachine': 0.8, 'microwave': 1.1,
}


def createScene(rootNode):
    alarmDist = 0.02
    contactDist = 0.02

    input_folder = config.PROCESSED_SEQS_ROOT
    subjects = sorted(os.listdir(input_folder))

    for sid in subjects:
        print(sid)
        sid_path = os.path.join(input_folder, sid)
        if not os.path.isdir(sid_path):
            continue
        for file_name in os.listdir(sid_path):
            name, _ = os.path.splitext(file_name)
            parts = name.split("_")
            object = parts[0]
            intent = parts[1]
            ID = "_".join(parts[2:])

            output_dir = os.path.join(config.FORCE_ROOT, sid, f'{object}_{intent}_{ID}_object')
            output_dir_left = os.path.join(config.FORCE_ROOT, sid, f'{object}_{intent}_{ID}_left')
            output_dir_right = os.path.join(config.FORCE_ROOT, sid, f'{object}_{intent}_{ID}_right')

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(output_dir_left, exist_ok=True)
            os.makedirs(output_dir_right, exist_ok=True)

            hand_vertices = 778
            object_vertices = len(trimesh.load(os.path.join(config.MESH_ROOT, object, "mesh.obj")).vertices)

            rootNode = Sofa.Core.Node("root")

            rootNode.addObject('VisualStyle', displayFlags='showVisual showWireframe showInteractionForceFields showForceFields')
            rootNode.addObject('MeshOBJLoader', name='leftHandLoader', filename=os.path.join(FIRST_FRAME_ROOT, sid, f'{object}_{intent}_{ID}_left.obj'))
            rootNode.addObject('MeshOBJLoader', name='rightHandLoader', filename=os.path.join(FIRST_FRAME_ROOT, sid, f'{object}_{intent}_{ID}_right.obj'))
            rootNode.addObject('MeshOBJLoader', name='objectLoader', filename=os.path.join(FIRST_FRAME_ROOT, sid, f'{object}_{intent}_{ID}_object.obj'))

            data = np.load(
                os.path.join(config.PROCESSED_SEQS_ROOT, sid, f'{object}_{intent}_{ID}.npy'),
                allow_pickle=True
            ).item()
            left_hand_verts_sequence = data['cam_coord']['verts.left'][:, 0, :, :]
            right_hand_verts_sequence = data['cam_coord']['verts.right'][:, 0, :, :]
            object_verts_sequence = data['cam_coord']['verts.object'][:, 0, :, :]
            num_frames = left_hand_verts_sequence.shape[0]

            dt = 0.01
            rootNode.dt = dt
            rootNode.gravity = [0, -9.81, 0]

            leftHandNode = rootNode.addChild('LeftHand')
            leftHandNode.addObject('EulerImplicitSolver')
            leftHandNode.addObject('CGLinearSolver', iterations=200, tolerance=1e-09, threshold=1e-09)

            leftHandNode.addObject('TriangleSetTopologyContainer', name='topo', src='@../leftHandLoader')
            leftHandNode.addObject('MechanicalObject', name='leftHandMO', template='Vec3d', showObject=1)
            leftHandNode.addObject("UniformMass", totalMass=0.5)

            leftHandNodeVisual = leftHandNode.addChild('LeftVisual')
            leftHandNodeVisual.addObject('OglModel',
                name='leftHandVisualModel',
                src='@../../leftHandLoader',
                color=[0.8, 0.7, 0.1, 1.0])

            leftHandNodeVisual.addObject('BarycentricMapping', name='LeftVisualMapping',
                input='@../leftHandMO',
                output='@leftHandVisualModel')

            leftCollision = leftHandNode.addChild('Collision')
            leftCollision.addObject('MeshTopology', src='@../../leftHandLoader')
            leftCollision.addObject('MechanicalObject', name='LeftStoringForces', template='Vec3d', showObject=1)
            leftCollision.addObject('PointCollisionModel', name='LeftCollisionModel', selfCollision=False)
            leftCollision.addObject('LineCollisionModel', name='LeftCollisionModelLine', selfCollision=False)
            leftCollision.addObject('TriangleCollisionModel', name='LeftCollisionModelTriangle', selfCollision=False)
            leftCollision.addObject('BarycentricMapping', name='LeftCollisionMapping',
                input='@../',
                output='@LeftStoringForces')

            rightHandNode = rootNode.addChild('RightHand')
            rightHandNode.addObject('EulerImplicitSolver')
            rightHandNode.addObject('CGLinearSolver', iterations=200, tolerance=1e-09, threshold=1e-09)

            rightHandNode.addObject('TriangleSetTopologyContainer', name='topo', src='@../rightHandLoader')
            rightHandNode.addObject('MechanicalObject', name='rightHandMO', template='Vec3d', showObject=1)
            rightHandNode.addObject("UniformMass", totalMass=0.5)

            rightHandNodeVisual = rightHandNode.addChild('RightVisual')
            rightHandNodeVisual.addObject('OglModel',
                name='rightHandVisualModel',
                src='@../../rightHandLoader',
                color=[0.8, 0.7, 0.1, 1.0])

            rightHandNodeVisual.addObject('BarycentricMapping', name='RightVisualMapping',
                input='@../rightHandMO',
                output='@rightHandVisualModel')

            rightCollision = rightHandNode.addChild('Collision')
            rightCollision.addObject('MeshTopology', src='@../../rightHandLoader')
            rightCollision.addObject('MechanicalObject', name='RightStoringForces', template='Vec3d', showObject=1)
            rightCollision.addObject('PointCollisionModel', name='RightCollisionModel', selfCollision=False)
            rightCollision.addObject('LineCollisionModel', name='RightCollisionModelLine', selfCollision=False)
            rightCollision.addObject('TriangleCollisionModel', name='RightCollisionModelTriangle', selfCollision=False)
            rightCollision.addObject('BarycentricMapping', name='RightCollisionMapping',
                input='@../',
                output='@RightStoringForces')

            object_mass = OBJECT_MASS[object]

            objectNode = rootNode.addChild('Object')
            objectNode.addObject('EulerImplicitSolver')
            objectNode.addObject('CGLinearSolver', iterations=200, tolerance=1e-09, threshold=1e-09)

            objectNode.addObject('TriangleSetTopologyContainer', name='topo', src='@../objectLoader')
            objectNode.addObject('MechanicalObject', name='objectMO', template='Vec3d', showObject=1)
            objectNode.addObject("UniformMass", totalMass=object_mass)

            objectNodeVisual = objectNode.addChild('ObjectVisual')
            objectNodeVisual.addObject('OglModel',
                name='objectVisualModel',
                src='@../../objectLoader',
                color=[0.2, 0.8, 0.5, 1.0])

            objectNodeVisual.addObject('BarycentricMapping', name='ObjectVisualMapping',
                input='@../objectMO',
                output='@objectVisualModel')

            objectCollision = objectNode.addChild('Collision')
            objectCollision.addObject('MeshTopology', src='@../../objectLoader')
            objectCollision.addObject('MechanicalObject', name='ObjectStoringForces', template='Vec3d', showObject=1)
            objectCollision.addObject('PointCollisionModel', name='ObjectCollisionModel', selfCollision=False)
            objectCollision.addObject('LineCollisionModel', name='ObjectCollisionModelLine', selfCollision=False)
            objectCollision.addObject('TriangleCollisionModel', name='ObjectCollisionModelTriangle', selfCollision=False)
            objectCollision.addObject('BarycentricMapping', name='ObjectCollisionMapping',
                input='@../objectMO',
                output='@ObjectStoringForces')

            print('READY ')

            class MeshSequenceController(Sofa.Core.Controller):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.leftHandMO = leftHandNode.getObject("leftHandMO")
                    self.rightHandMO = rightHandNode.getObject("rightHandMO")
                    self.objectMO = objectNode.getObject("objectMO")
                    self.leftHandForcesMO = leftHandNode.getObject('leftHandForcesMO')
                    self.rightHandForcesMO = rightHandNode.getObject('rightHandForcesMO')
                    self.left_hand_data = left_hand_verts_sequence
                    self.right_hand_data = right_hand_verts_sequence
                    self.object_data = object_verts_sequence
                    self.num_frames = num_frames
                    self.current_frame = 0
                    self.rootNode = rootNode
                    self.obj_storing_forces = objectCollision.getObject("ObjectStoringForces")
                    self.left_storing_forces = leftCollision.getObject("LeftStoringForces")
                    self.right_storing_forces = rightCollision.getObject("RightStoringForces")

                    self.contact_listener = kwargs['listener']
                    self.contact_listener_right = kwargs['listener_right']

                def onAnimateBeginEvent(self, event):
                    if self.current_frame > self.num_frames - 1:
                        self.stopped = True

                def onAnimateEndEvent(self, event):
                    if getattr(self, 'stopped', False):
                        return

                    self.leftHandMO.position.value = self.left_hand_data[self.current_frame]
                    self.rightHandMO.position.value = self.right_hand_data[self.current_frame]
                    self.objectMO.position.value = self.object_data[self.current_frame]

                    filtered_object_force = self.obj_storing_forces.force.value
                    filtered_left_force = self.left_storing_forces.force.value
                    filtered_right_force = self.right_storing_forces.force.value

                    np.save(os.path.join(output_dir, f'forces_{self.current_frame:04d}.npy'), filtered_object_force)
                    np.save(os.path.join(output_dir_left, f'forces_{self.current_frame:04d}.npy'), filtered_left_force)
                    np.save(os.path.join(output_dir_right, f'forces_{self.current_frame:04d}.npy'), filtered_right_force)

                    if self.current_frame < self.num_frames - 1:
                        self.current_frame += 1
                    else:
                        self.stopped = True
                        print("Mesh sequence finished.")

            rootNode.addObject("GenericConstraintSolver", name="constraintSolver", maxIterations=1000, tolerance=1e-5)
            rootNode.addObject('FreeMotionAnimationLoop')
            rootNode.addObject('EulerImplicitSolver', name='odesolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)
            rootNode.addObject('CGLinearSolver', iterations=200, tolerance=1e-09, threshold=1e-09)
            rootNode.addObject('CollisionPipeline', name='collision', verbose=1)
            rootNode.addObject('BruteForceDetection')
            rootNode.addObject('BVHNarrowPhase')
            rootNode.addObject('MinProximityIntersection', alarmDistance=alarmDist, contactDistance=contactDist, listening=True)
            rootNode.addObject('DefaultContactManager', name='ContactManager', response='PenalityContactForceField')

            listener = rootNode.addObject(
                "ContactListener",
                listening=True,
                collisionModel1="@LeftHand/Collision/LeftCollisionModel",
                collisionModel2="@Object/Collision/ObjectCollisionModel",
                name="myListener"
            )

            listener_right = rootNode.addObject(
                "ContactListener",
                listening=True,
                collisionModel1="@RightHand/Collision/RightCollisionModel",
                collisionModel2="@Object/Collision/ObjectCollisionModel",
                name="myListener_right"
            )

            controller = MeshSequenceController(
                name="meshSequenceController",
                listener=listener,
                listener_right=listener_right,
            )

            rootNode.addObject(controller)

            Sofa.Simulation.init(rootNode)

            while not getattr(controller, 'stopped', False):
                Sofa.Simulation.animate(rootNode, float(rootNode.dt.value))

    return rootNode
