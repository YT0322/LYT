import bpy


class ModalOperator(bpy.types.Operator):
    bl_idname = "object.modal_operator"
    bl_label = "Simple Modal Operator"

    def __init__(self):
        print("Start")

    def __del__(self):
        print("End")

    def execute(self, context):
        context.object.location.x = self.value_x / 100.0
        context.object.location.y = self.value_y / 100.0
        return {'FINISHED'}

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':  # Apply
            self.value_x = event.mouse_x
            self.value_y = event.mouse_y
            self.execute(context)
        elif event.type == 'LEFTMOUSE':  # Confirm
            return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:  # Cancel
            context.object.location.x = self.init_loc_x
            context.object.location.y = self.init_loc_y
            context.object.location.z = self.init_loc_z
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.init_loc_x = context.object.location.x
        self.init_loc_y = context.object.location.y
        self.init_loc_z = context.object.location.z
        self.value_x = event.mouse_x
        self.value_y = event.mouse_y
        self.execute(context)

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


bpy.utils.register_class(ModalOperator)

# test call
bpy.ops.object.modal_operator('INVOKE_DEFAULT')