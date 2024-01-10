import bpy
import math


def distance(temp):
    res = math.sqrt(
        math.pow(
            temp.x -
            loc1.x,
            2) +
        math.pow(
            temp.y -
            loc1.y,
            2) +
        math.pow(
            temp.z -
            loc1.z,
            2)
    )
    return res


# 获得中间牙齿的坐标位置
obj_center = bpy.data.objects.get("Baseline")  # 牙齿
loc1 = obj_center.original.location.copy()  # 浅拷贝

# 给每一个牙齿都增加距离约束
for j in range(1, 15, 1):
    obj = bpy.data.objects["crown_" + str(j)]

    bpy.ops.object.select_all(action="DESELECT")  # 全不选
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')  # 给每个牙齿设置好重心

    dis = distance(obj.original.location.copy())
    print("dis", dis)

    bpy.ops.object.constraint_add(type='LIMIT_DISTANCE')  # 增加约束
    bpy.context.object.constraints["Limit Distance"].target = bpy.data.objects["Baseline"]  # 添加距离约束
    bpy.context.object.constraints["Limit Distance"].distance = dis + 5

    # 添加”刚体“性质
    bpy.ops.rigidbody.object_add()

# bpy.context.active_object.name