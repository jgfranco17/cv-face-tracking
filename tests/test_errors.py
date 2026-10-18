from src.core.errors import ClientError, CoreError


class TestCoreError:
    def test_formats_message_without_additional_info(self) -> None:
        error = CoreError(category="detector", message="boom")

        assert error.category == "detector"
        assert error.message == "boom"
        assert error.additional_info == ""
        assert str(error) == "[DETECTOR] boom"

    def test_formats_message_with_additional_info(self) -> None:
        error = CoreError(
            category="camera",
            message="device failure",
            additional_info="camera index 0 is unavailable",
        )

        assert str(error) == (
            "[CAMERA] device failure\n\n"
            "Additional information: camera index 0 is unavailable"
        )


class TestClientError:
    def test_sets_client_category(self) -> None:
        error = ClientError("unable to start")

        assert isinstance(error, CoreError)
        assert error.category == "CLIENT"
        assert str(error) == "[CLIENT] unable to start"
