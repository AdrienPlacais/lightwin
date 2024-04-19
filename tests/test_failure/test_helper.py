import pytest

from failures.helper import whole_k_out_of_n


@pytest.mark.implementation
class TestStrategy:
    """Test the different strategies."""

    my_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    def test_k_out_of_n_down_single_fail(self) -> None:
        """Check that our sorting works."""
        k = 5
        given = whole_k_out_of_n(
            self.my_list, main_items=["4"], k=k, tie_politics="upstream first"
        )
        expected = ["3", "5", "2", "6", "1"]
        assert given == expected

    def test_k_out_of_n_up_single_fail(self) -> None:
        """Check that our sorting works."""
        k = 5
        given = whole_k_out_of_n(
            self.my_list,
            main_items=["4"],
            k=k,
            tie_politics="downstream first",
        )
        expected = ["5", "3", "6", "2", "7"]
        assert given == expected
