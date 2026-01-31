
from sdks.novavision.src.helper.package import PackageHelper
from components.DrawBoundingRectangle.src.models.PackageModel import PackageConfigs, ConfigExecutor,PackageModel,OutputImage,DrawBboxOutputs,DrawBboxExecutor,DrawBboxResponse


def build_response(context):
    output_image = OutputImage(value=context.image)
    detect_outputs = DrawBboxOutputs(outputImage=output_image)
    draw_bbox_response = DrawBboxResponse(outputs=detect_outputs)
    draw_bbox_executor = DrawBboxExecutor(value=draw_bbox_response)
    executor = ConfigExecutor(value=draw_bbox_executor)
    package_configs = PackageConfigs(executor=executor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    packageModel = package.build_model(context)
    return packageModel