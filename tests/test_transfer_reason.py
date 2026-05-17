import unittest

from agent.handlers import sanitize_transfer_reason_token


class TransferReasonTests(unittest.TestCase):
    def test_known_coverage_reason_is_preserved(self):
        self.assertEqual(
            sanitize_transfer_reason_token("phone_coverage_off"),
            "phone_coverage_off",
        )

    def test_unknown_reason_is_ignored(self):
        self.assertEqual(sanitize_transfer_reason_token("forged_reason"), "")
        self.assertEqual(sanitize_transfer_reason_token("phone_coverage_off\nextra"), "")


if __name__ == "__main__":
    unittest.main()
