from text_localizer.cli import build_parser


def test_required_commands_are_registered() -> None:
    parser = build_parser()
    command_arguments = {
        "doctor": ["doctor"],
        "inspect-video": ["inspect-video", "placeholder"],
        "extract-frames": ["extract-frames", "placeholder"],
        "analyze-image": ["analyze-image", "placeholder"],
        "setup-models": ["setup-models", "--ocr"],
    }
    for command, arguments in command_arguments.items():
        namespace = parser.parse_args(arguments)
        assert namespace.command == command
