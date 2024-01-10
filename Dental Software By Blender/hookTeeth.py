import bpy
import math
from math import radians
import mathutils
import bmesh
from mathutils import Vector


# bpy.ops.object.mode_set(mode="EDIT")
def distance(temp):
    res = math.sqrt(
        math.pow(
            temp.co.x -
            loc1.x,
            2) +
        math.pow(
            temp.co.y -
            loc1.y,
            2) +
        math.pow(
            temp.co.z -
            loc1.z,
            2)
    )
    return res


obj1 = bpy.data.objects.get("crown_1")
loc1 = obj1.original.location.copy()  # 浅拷贝
# loc1[2] += 6
print("obj1:", loc1)

# find all points in the range
obj_gum = bpy.data.meshes.get("lc_UpperJawScan")
# 遍历牙龈上的所有点
# 用字典存储 点 及其对应的weight
my_dict = {}  # 牙龈上的点
r = 10
bpy.ops.object.mode_set(mode="OBJECT")
# 取消物体的ACTIVE性质
obj = bpy.data.objects.get("lc_UpperJawScan")
if obj:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

bpy.ops.object.mode_set(mode="EDIT")

bpy.ops.object.vertex_group_set_active(group="line")
bpy.ops.object.vertex_group_select()  # 选中牙龈线上的点
gum_line = []
num = 0
bm = bmesh.from_edit_mesh(bpy.context.active_object.data)
for m in bm.verts:
    if m.select:
        gum_line.append(m)
# gum_line = []
# for m in bpy.context.active_object.data.vertices:
#    if m.select:
#        gum_line.append(m)

# bpy.ops.object.mode_set(mode="OBJECT")
## 取消物体的ACTIVE性质
# obj = bpy.data.objects.get("lc_UpperJawScan_label_gum")
# if obj:
#    bpy.ops.object.select_all(action="DESELECT")
#    obj.select_set(True)
#    bpy.context.view_layer.objects.active = obj

# bpy.ops.object.mode_set(mode="EDIT")
num = 0

for i in obj_gum.vertices:
    # 先排除一部分点
    if (i.co.x >= r and i.co.y >= r and i.co.z >= r):
        continue

    result = distance(i)

    if (r >= result):
        num += 1
        # print(i.co)
        # 将符合条件的向量写入字典 做key
        flag = 1
        for ii in gum_line:  # 排除钩挂上的点
            if (distance(ii) == distance(i)):
                flag = 0;
                print("in")
                break

        if (flag == 1):
            my_dict[i] = distance(i)
        # my_dict[i] = 1.0 * (math.sin(result))
# 输出字典中的内容
# for key, value in my_dict.items():
#    print(key, value)
print("num", num)
print("gum_line", len(gum_line))
print("len", len(my_dict))

# 牙龈上的点到中点的最短距离
min = 9999
for value in my_dict.values():
    if (min > value):
        min = value

# 牙齿向x轴方向移动一个单位
unit = -1
obj1.location[0] += unit
#
# obj1.rotation_euler[0] += radians(45)
loc2 = obj1.original.location  # 现在的位置
# loc2 = obj1.original.location+mathutils.Vector((0,0,6))

# 捕捉牙齿中点的运动方向
mov = loc2 - loc1

# 移动范围内的点
# for key, value in my_dict.items():
##    key.co += mov * math.cos(0.7 - value)
#    key.co += mov * r/min *(r - value)/r

for line in gum_line:
    line.co += mov
distance = {}
bpy.ops.object.mode_set(mode="OBJECT")
for key, value in my_dict.items():
    #    key.co += mov * math.cos(0.7 - value)
    key.co += mov * r / min * (r - value) / r


# 在OBJECT模式下才能移动
#for key, value in distance.items():
#    key.co += value

# 平滑表面
bpy.ops.object.shade_smooth()