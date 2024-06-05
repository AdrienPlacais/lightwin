"""Test the strategy and helper functions to set compensating cavities.

.. todo::
    More clean with mocking ListOfElements

"""

from functools import partial

import pytest

from failures.helper import gather
from failures.strategy import k_out_of_n, l_neighboring_lattices
from util.helper import chunks


@pytest.mark.smoke
class TestStrategy:
    """Test the different strategies."""

    my_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    my_nested_list = [x for x in chunks(my_list, n_size=2)]

    def test_k_out_of_n_down_single_fail(self) -> None:
        """Check that our sorting works."""
        k = 5
        failed_elements = ["4"]
        obtained = k_out_of_n(
            self.my_list,
            failed_elements=failed_elements,
            k=k,
            tie_politics="upstream first",
            remove_failed=False,
        )
        expected = failed_elements + ["3", "5", "2", "6", "1"]
        assert obtained == expected, f"{obtained = } but {expected = }"

    def test_k_out_of_n_up_single_fail(self) -> None:
        """Check that our sorting works."""
        k = 5
        failed_elements = ["4"]
        obtained = k_out_of_n(
            self.my_list,
            failed_elements=failed_elements,
            k=k,
            tie_politics="downstream first",
            remove_failed=False,
        )
        expected = failed_elements + ["5", "3", "6", "2", "7"]
        assert obtained == expected, f"{obtained = } but {expected = }"

    def test_k_out_of_n_full_lattice_failed(self) -> None:
        """Check selection for several failed cavities."""
        k = 3
        failed_elements = ["4", "5"]
        obtained = k_out_of_n(
            self.my_list,
            failed_elements=failed_elements,
            k=k,
            tie_politics="upstream first",
            remove_failed=False,
        )
        expected = failed_elements + ["3", "6", "2", "7", "1", "8"]
        assert obtained == expected, f"{obtained = } but {expected = }"

    def test_k_out_of_n_several_elements_far_away(self) -> None:
        """Check selection for several non-contiguous failed cavities."""
        k = 3
        failed_elements = ["4", "8"]
        obtained = k_out_of_n(
            self.my_list,
            failed_elements=failed_elements,
            k=k,
            tie_politics="upstream first",
            remove_failed=False,
        )
        expected = failed_elements + ["3", "5", "7", "9", "2", "6"]
        assert obtained == expected, f"{obtained = } but {expected = }"

    def test_k_out_of_n_several_elements_close(self) -> None:
        """Check selection for several non-contiguous failed cavities."""
        k = 3
        failed_elements = ["4", "6"]
        obtained = k_out_of_n(
            self.my_list,
            failed_elements=failed_elements,
            k=k,
            tie_politics="upstream first",
            remove_failed=False,
        )
        expected = failed_elements + ["3", "5", "7", "2", "8", "1"]
        assert obtained == expected, f"{obtained = } but {expected = }"

    def test_overlapping(self) -> None:
        """Check selection for when several failures need same compensating."""
        k = 2
        failed_elements = ["2", "7", "8"]
        fun_sort = partial(
            k_out_of_n,
            elements=self.my_list,
            k=k,
            tie_politics="upstream first",
            remove_failed=False,
        )
        obtained = gather(failed_elements=failed_elements, fun_sort=fun_sort)  # type: ignore

        expected = ([["2"], ["7", "8"]], [["1", "3"], ["6", "9", "5", "10"]])
        assert obtained == expected, f"{obtained = } but {expected = }"

    def test_l_neighboring_lattices_up_single_fail(self) -> None:
        """Check selection of the l neighboring lattices."""
        l = 3
        failed_elements = ["4"]

        obtained = l_neighboring_lattices(
            self.my_nested_list,
            failed_elements=failed_elements,
            l=l,
            tie_politics="upstream first",
            remove_failed=False,
        )
        expected = failed_elements + ["5", "2", "3", "6", "7", "0", "1"]
        assert obtained == expected, f"{obtained = } but {expected = }"

    def test_l_neighboring_lattices_down_single_fail(self) -> None:
        """Check selection of the l neighboring lattices."""
        l = 3
        failed_elements = ["4"]

        obtained = l_neighboring_lattices(
            self.my_nested_list,
            failed_elements=failed_elements,
            l=l,
            tie_politics="downstream first",
            remove_failed=False,
        )
        expected = failed_elements + ["5", "6", "7", "2", "3", "8", "9"]
        assert obtained == expected, f"{obtained = } but {expected = }"

    def test_l_neighboring_lattices_two_faults_same_latt(self) -> None:
        """Check selection of the l neighboring lattices."""
        l = 3
        failed_elements = ["4", "5"]

        obtained = l_neighboring_lattices(
            self.my_nested_list,
            failed_elements=failed_elements,
            l=l,
            tie_politics="upstream first",
            remove_failed=False,
        )
        expected = failed_elements + ["2", "3", "6", "7", "0", "1"]
        assert obtained == expected, f"{obtained = } but {expected = }"

    def test_l_neighboring_lattices_two_faults_diff_latt(self) -> None:
        """Check selection of the l neighboring lattices."""
        l = 2
        failed_elements = ["4", "7"]

        obtained = l_neighboring_lattices(
            self.my_nested_list,
            failed_elements=failed_elements,
            l=l,
            tie_politics="upstream first",
            remove_failed=False,
        )
        #           [fail---] [fail---] [comp1-]  [comp2-]  [comp1-]  ['comp2']
        expected = ["4", "5", "6", "7", "2", "3", "8", "9", "0", "1", "10"]
        assert obtained == expected, f"{obtained = } but {expected = }"

    def test_l_neighboring_lattices_min_number(self) -> None:
        """Check that if we remove a cav in a lattice, it is skipped."""
        l = 3
        failed_elements = ["4"]
        nested_list = [x for x in self.my_nested_list if "6" not in x]

        obtained = l_neighboring_lattices(
            nested_list,
            failed_elements=failed_elements,
            l=l,
            tie_politics="upstream first",
            remove_failed=False,
            min_number_of_cavities_in_lattice=1,
        )
        expected = failed_elements + ["5", "2", "3", "8", "9", "0", "1"]
        assert obtained == expected, f"{obtained = } but {expected = }"

    def test_positive_penalty(self) -> None:
        """Check functionality to unbalance the compensating cavities."""
        k = 6
        failed_elements = ["4"]

        obtained = k_out_of_n(
            self.my_list,
            failed_elements=failed_elements,
            k=k,
            tie_politics="downstream first",
            shift=+2,
            remove_failed=False,
        )
        expected = failed_elements + ["5", "6", "7", "3", "8", "2"]
        assert obtained == expected, f"{obtained = } but {expected = }"

    def test_negative_penalty(self) -> None:
        """Check functionality to unbalance the compensating cavities."""
        k = 6
        failed_elements = ["4"]

        obtained = k_out_of_n(
            self.my_list,
            failed_elements=failed_elements,
            k=k,
            tie_politics="upstream first",
            shift=-2,
            remove_failed=False,
        )
        expected = failed_elements + ["3", "2", "1", "5", "0", "6"]
        assert obtained == expected, f"{obtained = } but {expected = }"
