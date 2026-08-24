from text_localizer.cli import build_parser


def test_required_commands_are_registered() -> None:
    parser = build_parser()
    for command in ("doctor", "inspect-video", "extract-frames", "analyze-image"):
        arguments = [command] if command == "doctor" else [command, "placeholder"]
        namespace = parser.parse_args(arguments)
        assert namespace.command == command
