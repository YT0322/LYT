import bpy
import math
import bmesh
from math import radians
import mathutils

from mathutils import Vector


# bpy.ops.object.mode_set(mode="EDIT")
def distance(temp, object):
    res = math.sqrt(
        math.pow(
            temp.co.x -
            object.x,
            2) +
        math.pow(
            temp.co.y -
            object.y,
            2) +
        math.pow(
            temp.co.z -
            object.z,
            2)
    )
    return res


def distance_2(temp1, temp2):
    # 只算投影面积
    res = math.sqrt(
        math.pow(
            temp1.co.x -
            temp2.co.x,
            2) +
        math.pow(
            temp1.co.y -
            temp2.co.y,
            2)
        +
        math.pow(
            temp1.co.z -
            temp2.co.z,
            2)
    )
    return res


obj1 = bpy.data.objects.get("crown_1")
loc1 = obj1.original.location.copy()  # 浅拷贝

obj = bpy.data.objects.get("lc_UpperJawScan")
# find all points in the range
obj_gum = bpy.data.meshes.get("lc_UpperJawScan")
# 遍历牙龈上的所有点
# 用字典存储 点 及其对应的weight
my_dict = {}  # 牙龈上的点
nearest = {}  # 牙龈点 对应 牙龈线上的点
nearest_dis = {}  # 牙龈点 到 对应 牙龈线上的点的距离
gum_loc = {}  # 牙龈上的点 与其 初始位置

r = 10

# 找出牙龈线上的点
bpy.ops.object.mode_set(mode="OBJECT")
# 取消物体的ACTIVE性质

# print("obj" , obj)
if obj:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

bpy.ops.object.mode_set(mode="EDIT")

bpy.ops.object.vertex_group_set_active(group="line")
bpy.ops.object.vertex_group_select()  # 选中牙龈线上的点
# gum_line = [i for i in bpy.context.active_object.data.vertices if i.select]
gum_line = []
num = 0
bm = bmesh.from_edit_mesh(bpy.context.active_object.data)
for m in bm.verts:
    if m.select:
        gum_line.append(m)

# bpy.ops.object.mode_set(mode="OBJECT")
## 取消物体的ACTIVE性质
# obj = bpy.data.objects.get("lc_UpperJawScan")
# if obj:
#    bpy.ops.object.select_all(action="DESELECT")
#    obj.select_set(True)
#    bpy.context.view_layer.objects.active = obj

# bpy.ops.object.mode_set(mode="EDIT")

# 填充gum_loc：牙龈上的点 与其 初始位置
for i in gum_line:
    gum_loc[i] = i.co.copy()

# print("obj_gum" , obj_gum)

# 填充my_dict：牙龈上的点 球心的距离
for i in obj_gum.vertices:
    # 先排除一部分点
    if (i.co.x >= r and i.co.y >= r and i.co.z >= r):
        continue

    result = distance(i, loc1)
    if (r >= result):
        # 将符合条件的向量写入字典 做key
        flag = 1
        for ii in gum_line:  # 排除牙龈线上的点
            if (distance(ii , loc1) == distance(i , loc1)):
                flag = 0;
                break

        if (flag == 1):
            my_dict[i] = distance(i, loc1)

print("len", len(my_dict))
# 填充nearest：牙龈上的点 对应的牙龈线上的最近点
# 填充nearest_dis：牙龈上的点 对应的牙龈线上的最近点 的最近距离
temp_vertex = Vector((0, 0, 0))
for key, value in my_dict.items():
    min = 9999
    for m in gum_line:
        result = distance_2(key, m)
        if (result < min):
            min = result
            temp_vertex = m
    nearest[key] = temp_vertex
    nearest_dis[key] = min

#字典nearest_oppo存 牙龈线上的点对应的所有的牙齿上的点
nearest_oppo = {}
count = 0
for key, value in nearest.items():
    #todo
    flag = 0
    for key1 in nearest_oppo.keys():
        if(value == key1):
            flag = 1
            break

    if(flag == 0):
        list = []
        list.append(key)
        nearest_oppo[value] = list

    else:
        nearest_oppo[value].append(key)


# 牙龈上的点到牙龈线上的最长距离
gum_longest = {}
# for k in gum_line:
#     for key, value in nearest_oppo.items():
#         max = 0
#         if (key == k):
#             if(max < nearest_dis[key]):
#                 max = nearest_dis[key]
#                 gum_longest[k] = max
for key, value in nearest_oppo.items():
    max = 0
    for i in range(len(value)):
        if(max < distance_2(value[i] , key)):
            max = distance_2(value[i] , key)
    gum_longest[key] = max
# print("nearest", len(nearest))
# print("nearest_dis", len(nearest_dis))
# print("gum_longest", gum_longest)

# 牙齿向x轴方向移动一个单位
unit = -1
obj1.location[0] += unit

# obj1.rotation_euler[0] += radians(45)
loc2 = obj1.original.location  # 现在的位置

mov = loc2 - loc1

#num = 0
for line in gum_line:
    line.co += mov

distance = {}  #
for key, value in my_dict.items():
    # 对应的牙龈线上的点 移动的距离
    #    bpy.ops.object.mode_set(mode="EDIT")
    gum = nearest[key]
    mov = gum.co - gum_loc[gum]
    r = gum_longest[gum]
    if(r != 0):
        distance[key] = mov * nearest_dis[key] / r

#    print( "after" , key.co )

bpy.ops.object.mode_set(mode="OBJECT")
# 在OBJECT模式下才能移动
for key, value in distance.items():
    key.co += value
print("len1" , len(distance))
print("len2" , len(my_dict))
# 平滑表面
bpy.ops.object.shade_smooth()