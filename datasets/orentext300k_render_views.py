import bpy
import math
import shutil
import argparse
from mathutils import Vector, Euler

focus = 50  # Focal length in mm
width = 1024  # Image width in pixels
height = 768  # Image height in pixels
sensor_width = 36  # Sensor width in mm
sensor_height = 27  # Sensor width in mm


def render(model_path, save_path):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = 'CUDA'
    bpy.context.scene.cycles.samples = 16  # Set the sample number to 16
    bpy.context.scene.cycles.use_denoising = True  # Turn on noise reduction
    bpy.context.scene.cycles.denoising_strength = 0.2  # Set noise reduction intensity to 0.2

    obj = bpy.data.objects.get('Cube')
    if obj:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.delete()

    # Create a set
    if "Import" not in bpy.data.collections:
        collection = bpy.data.collections.new("Import")
        bpy.context.scene.collection.children.link(collection)
    else:
        collection = bpy.data.collections["Import"]

    id = model_path.split('/')[-1].split('.')[0]

    # Import .glb file
    bpy.ops.import_scene.gltf(filepath=model_path)

    # Place the imported object into the set
    for obj in bpy.context.selected_objects:
        bpy.data.collections["Import"].objects.link(obj)
        bpy.context.scene.collection.objects.unlink(obj)

    # Select all objects within the collection.
    bpy.ops.object.select_all(action='DESELECT')
    for obj in collection.objects:
        obj.select_set(True)

    # Implementation instance
    bpy.ops.object.make_single_user(object=True, obdata=True, material=False, animation=False)

    # Disconnect the parent-child level and maintain the transformation result
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    # Delete all empty objects
    empty_objects = [obj for obj in bpy.data.objects if obj.type == 'EMPTY']
    for empty_obj in empty_objects:
        bpy.data.objects.remove(empty_obj, do_unlink=True)

    # Select all non-grid objects in the set and delete
    bpy.ops.object.select_all(action='DESELECT')
    for obj in collection.objects:
        if obj.type != 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete(use_global=False)

    # Select all grid objects and merge
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    selected_objects = bpy.context.selected_objects

    if len(selected_objects) > 1:
        bpy.context.view_layer.objects.active = selected_objects[0]
        bpy.ops.object.join()

    # Traverse all grid objects in the scene
    merged_object = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            merged_object = obj
            break

    # If the merged object does not exist, clear the set and skip the file.
    if merged_object is None:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in collection.objects:
            obj.select_set(True)
        bpy.ops.object.delete(use_global=False)
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        print(f"Skip file {model_path}: The merged object does not exist.")
        return

    # Set the origin to the geometric center
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    merged_object.location = (0, 0, 0)

    # Apply all transformations
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Calculate maximum size
    try:
        dimensions = merged_object.dimensions
        max_dim = max(dimensions.x, dimensions.y, dimensions.z)
    except Exception as e:
        print(f"Unable to obtain the size of {merged_object.name}: {e}")
        bpy.ops.object.select_all(action='DESELECT')
        for obj in collection.objects:
            obj.select_set(True)
        bpy.ops.object.delete(use_global=False)
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        return

    # Apply zoom
    scale_factor = 2.0 / max_dim
    merged_object.scale = (scale_factor, scale_factor, scale_factor)

    # Apply scaling transformation
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Rendering settings
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Add 8 large-area light sources
    light_positions = []
    num_lights = 8
    radius = 8

    # Create light source position
    for i in range(num_lights):
        angle = i * (2 * math.pi / num_lights)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z_values = [10, 5, 0, -5]
        for z in z_values:
            light_positions.append((x, y, z))

    for pos in light_positions:
        bpy.ops.object.light_add(type='AREA', location=pos)
        light = bpy.context.active_object
        light.data.energy = 400
        light.data.size = 5
        light.data.shape = 'RECTANGLE'
        light.data.shadow_soft_size = 6
        light.data.use_shadow = True

    # Set ambient light
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    bg_node.inputs[1].default_value = 8

    # Set camera position
    camera = bpy.data.objects['Camera']
    camera.location = (0, -5, 0)
    camera.rotation_euler = (math.pi / 2, 0, 0)
    camera.data.lens = focus
    camera.data.sensor_width = sensor_width
    camera.data.sensor_height = sensor_height

    # Set rendering parameters
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.film_transparent = True  # Start film transparency

    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    # Create input render layer node
    render_layers = nodes.new('CompositorNodeRLayers')
    # Create depth output nodes
    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.base_path = ''
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = 'OPEN_EXR'
    depth_file_output.format.color_depth = '16'

    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

    # Define object rotation
    view_angles = [
        (0, 0, 0),  # Front
        (Euler((0, 0, math.pi / 4), 'XYZ').to_quaternion() @ Euler((math.pi / 8, -math.pi / 8, 0),
                                                                   'XYZ').to_quaternion()).to_euler('XYZ'),  # Right front 45 degrees
        (0, 0, math.pi / 2),  # right side
        (Euler((0, 0, 3 * math.pi / 4), 'XYZ').to_quaternion() @ Euler((-math.pi / 8, -math.pi / 8, 0),
                                                                       'XYZ').to_quaternion()).to_euler('XYZ'),  # Left front 45 degrees
        (0, 0, math.pi),  # Back
        (Euler((0, 0, -3 * math.pi / 4), 'XYZ').to_quaternion() @ Euler((-math.pi / 8, math.pi / 8, 0),
                                                                        'XYZ').to_quaternion()).to_euler('XYZ'), # Left rear 45 degrees
        (0, 0, 3 * math.pi / 2),  # left side
        (Euler((0, 0, -math.pi / 4), 'XYZ').to_quaternion() @ Euler((math.pi / 8, math.pi / 8, 0),
                                                                    'XYZ').to_quaternion()).to_euler('XYZ'),  # Left front 45 degrees
        (math.pi / 2, 0, 0),  # Top
        (3 * math.pi / 2, 0, 0),  # Bottom
    ]
    # Standard View
    # view_angles = [
    #     (0, 0, 0),  # Front
    #     (0, 0, math.pi / 2),  # right side
    #     (0, 0, math.pi),  # Back
    #     (0, 0, 3 * math.pi / 2),  # left side
    #     (math.pi / 2, 0, 0),  # Top
    #     (3 * math.pi / 2, 0, 0),  # Bottom
    # ]
    # Random rotation
    # view_angles = [(random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi)) for _ in range(12)]

    for i, angle in enumerate(view_angles):
        # Rotate model to specified view angle
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = angle

        # From the camera to take pictures
        bpy.context.scene.camera = camera
        depth_file_output.file_slots[0].path = f"{save_path}/{i}"
        bpy.ops.render.render(write_still=True)

        # Save image
        output_filepath = f"{save_path}/temp.png"
        bpy.data.images['Render Result'].save_render(filepath=output_filepath)

        shutil.move(f"{save_path}/temp.png", f"{save_path}/{id}-{i}.png")

    # Delete the unique mesh in the scene.
    bpy.ops.object.select_all(action='DESELECT')
    merged_object.select_set(True)
    bpy.ops.object.delete(use_global=False)

    # Delete all grids at the file data level
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    # Clear unused data
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str)
    args.add_argument("--save_path", type=str)
    args = args.parse_args()
    render(args.model_path, args.save_path)
