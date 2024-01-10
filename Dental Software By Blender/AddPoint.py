import bpy


def AddPoint(name, loc, Diameter=1, CollName=None):
    obj = bpy.data.objects.get(name)
    if obj:
        bpy.data.objects.remove(obj)
    temp = bpy.data.meshes.get(name + "_mesh")
    if temp:
        bpy.data.meshes.remove(temp)
    bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=loc, scale=(1, 1, 1))
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.merge(type='CENTER')

    P = bpy.context.object
    P.name = name
    P.data.name = name + "_mesh"

    matName = f"{name}_Mat"
    # mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)

    return P


for j in range(1, 15, 1):
    obj = bpy.data.objects["crown_" + str(j)]

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")  # 全不选
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')  # 给每个牙齿设置好重心

    loc = obj.original.location.copy()
    AddPoint("point_" + str(j), loc)

# 将所有点合并成一个物体
for i in range(1, 15, 1):
    obj = bpy.data.objects["point_" + str(j)]

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")  # 全不选
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.join()
