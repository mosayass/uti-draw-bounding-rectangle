
from sdks.novavision.src.helper.package import PackageHelper
from components.DrawBoundingRectangle.src.models.PackageModel import PackageConfigs, ConfigExecutor,PackageModel,OutputImage,DrawBoundingRectangleOutputs,DrawBoundingRectangleExecutor,DrawBoundingRectangleResponse


def build_response(context):
    output_image = OutputImage(value=context.image)
    detect_outputs = DrawBoundingRectangleOutputs(outputImage=output_image)
    draw_BoundingRectangle_response = DrawBoundingRectangleResponse(outputs=detect_outputs)
    draw_BoundingRectangle_executor = DrawBoundingRectangleExecutor(value=draw_BoundingRectangle_response)
    executor = ConfigExecutor(value=draw_BoundingRectangle_executor)
    package_configs = PackageConfigs(executor=executor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    packageModel = package.build_model(context)
    return packageModel