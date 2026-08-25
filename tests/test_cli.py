from text_localizer.cli import build_parser


def test_required_commands_are_registered() -> None:
    parser = build_parser()
    command_arguments = {
        "doctor": ["doctor"],
        "inspect-video": ["inspect-video", "placeholder"],
        "extract-frames": ["extract-frames", "placeholder"],
        "analyze-image": ["analyze-image", "placeholder"],
        "deduplicate-frames": ["deduplicate-frames", "placeholder"],
        "index-video": ["index-video", "placeholder"],
        "localize-video": ["localize-video", "placeholder"],
        "render-tracks": ["render-tracks", "placeholder"],
        "setup-models": ["setup-models", "--ocr"],
        "render-video": ["render-video", "placeholder"],
    }
    for command, arguments in command_arguments.items():
        namespace = parser.parse_args(arguments)
        assert namespace.command == command
