// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "test_repo",
    products: [
        .executable(
            name: "test_repo",
            targets: ["test_repo"]),
    ],
    targets: [
        .executableTarget(
            name: "test_repo",
            dependencies: [],
            path: "src"),
    ]
)
