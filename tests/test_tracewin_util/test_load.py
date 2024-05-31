"""Test that the lines of the ``.dat`` are properly understood."""

import hashlib
from pathlib import Path
from pprint import pformat

from core.commands.adjust import Adjust
from tracewin_utils.load import load_dat_file, slice_dat_line


class TestSlice:
    """Test functions to convert a ``.dat`` line to list of arguments."""

    def test_basic_line(self) -> None:
        """Test that a basic line is properly sliced."""
        line = "DRIFT 76"
        expected = ["DRIFT", "76"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_line_with_more_arguments(self) -> None:
        line = "FIELD_MAP 100 5 0.9 0.7 54e4 3 65.6e10"
        expected = line.split()
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_basic_comment(self) -> None:
        """Test that a basic comment is properly sliced."""
        line = ";DRIFT 76"
        expected = [";", "DRIFT 76"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_basic_comment_with_space(self) -> None:
        """Test that a basic comment is properly sliced."""
        line = "; DRIFT 76"
        expected = [";", "DRIFT 76"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_element_with_a_name(self) -> None:
        """Test that a named element is properly sliced."""
        line = "Louise: DRIFT 76"
        expected = ["Louise", "DRIFT", "76"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_element_with_a_name_additional_space(self) -> None:
        """Test that a named element is properly sliced."""
        line = "Michel : DRIFT 76"
        expected = ["Michel", "DRIFT", "76"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_diagnostic_with_a_weight(self) -> None:
        """Test that a weighted element is properly sliced."""
        line = "DIAG_BONJOURE(1e3) 777 0 1 2"
        expected = ["DIAG_BONJOURE", "(1e3)", "777", "0", "1", "2"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_diagnostic_with_a_weight_additional_space(self) -> None:
        """Test that a weighted element is properly sliced."""
        line = "DIAG_BONJOURE (1e3) 777 0 1 2"
        expected = ["DIAG_BONJOURE", "(1e3)", "777", "0", "1", "2"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_diagnostic_with_a_weight_different_fmt(self) -> None:
        """Test that a weighted element is properly sliced."""
        line = "DIAG_BONJOURE (4.5) 777 0 1 2"
        expected = ["DIAG_BONJOURE", "(4.5)", "777", "0", "1", "2"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_multiple_semicommas(self) -> None:
        """Check that when we have several ;, only the first is kept."""
        line = ";;;;;;;; Section1: ;;;;;;;"
        expected = [line[0], line[1:]]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_comment_at_end_of_line_is_removed(self) -> None:
        """Test that EOL comments are removed to avoid any clash."""
        line = "DRIFT 76 ; this drift is where we put the coffee machine"
        expected = ["DRIFT", "76"]
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_line_with_nothing_but_spaces(self) -> None:
        """Test that empty line is correctly understood."""
        line = "    "
        expected = []
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"

    def test_windows_like_path(self) -> None:
        """Test that the : does not mess with the code."""
        line = "field_map_path C:\\path\\to\\field_maps\\"
        expected = line.split()
        returned = slice_dat_line(line)
        assert expected == returned, f"{returned = } but {expected = }"


def md5(fname: Path | str) -> str:
    """Give checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class TestLoadDatFile:
    """Ensure that the ``.dat`` file will be correctly loaded."""

    dat_path = Path("data/example/example.dat")
    expected_checksum = "06d55c23082cedd0cc8f065dc04e608d"
    expected_dat_filecontent = [
        ["FIELD_MAP_PATH", "field_maps_1D"],
        ["LATTICE", "10", "0"],
        ["FREQ", "352.2"],
        ["QUAD", "200", "5.66763", "100", "0", "0", "0", "0", "0", "0"],
        ["DRIFT", "150", "100", "0", "0", "0"],
        ["QUAD", "200", "-5.67474", "100", "0", "0", "0", "0", "0", "0"],
        ["DRIFT", "150", "100", "0", "0", "0"],
        ["DRIFT", "589.42", "30", "0", "0", "0"],
        [
            "FIELD_MAP",
            "100",
            "415.16",
            "153.171",
            "30",
            "0",
            "1.55425",
            "0",
            "0",
            "Simple_Spoke_1D",
            "0",
        ],
        ["DRIFT", "264.84", "30", "0", "0", "0"],
        [
            "FIELD_MAP",
            "100",
            "415.16",
            "156.892",
            "30",
            "0",
            "1.55425",
            "0",
            "0",
            "Simple_Spoke_1D",
            "0",
        ],
        ["DRIFT", "505.42", "30", "0", "0", "0"],
        ["DRIFT", "150", "100", "0", "0", "0"],
        ["QUAD", "200", "5.77341", "100", "0", "0", "0", "0", "0", "0"],
        ["DRIFT", "150", "100", "0", "0", "0"],
    ]
    idx_fm1 = 8
    idx_fm2 = 10
    line_adj_phase = ["ADJUST", "42", "3", "1", "-180", "180"]
    line_adj_ampl = ["ADJUST", "42", "6", "1", "1.50", "1.60"]

    def test_file_was_not_changed(self) -> None:
        """Compare checksums to verify file is still the same.

        Otherwise, I may mess up with those tests.

        """
        actual_checksum = md5(self.dat_path)
        assert actual_checksum == self.expected_checksum, (
            f"The checksum of {self.dat_path} does not match the expected one."
            " Maybe the file was edited?"
        )

    def test_some_lines_of_the_dat(self) -> None:
        """Check one some lines that the loading is correct."""
        actual_dat_filecontent = load_dat_file(self.dat_path)
        expected_dat_filecontent = self.expected_dat_filecontent
        assert expected_dat_filecontent == actual_dat_filecontent[:15], (
            f"Expected:\n{pformat(expected_dat_filecontent, width=120)}\nbut "
            f"returned:\n{pformat(actual_dat_filecontent[:15], width=120)}"
        )

    def test_insert_instruction(self) -> None:
        """Check that an instruction will be inserted at the proper place."""
        instruction_1 = Adjust(self.line_adj_phase, self.idx_fm1)

        expected_dat_filecontent = self.expected_dat_filecontent[:-1]
        expected_dat_filecontent.insert(self.idx_fm1, self.line_adj_phase)

        actual_dat_filecontent = load_dat_file(
            self.dat_path, instructions_to_insert=(instruction_1,)
        )
        assert expected_dat_filecontent == actual_dat_filecontent[:15], (
            f"Expected:\n{pformat(expected_dat_filecontent, width=120)}\nbut "
            f"returned:\n{pformat(actual_dat_filecontent[:15], width=120)}"
        )

    def test_insert_instructions(self) -> None:
        """Check that several instructions will work together."""
        instructions = (
            Adjust(self.line_adj_phase, self.idx_fm1),
            Adjust(self.line_adj_ampl, self.idx_fm1),
            Adjust(self.line_adj_phase, self.idx_fm2),
            Adjust(self.line_adj_ampl, self.idx_fm2),
        )
        expected_dat_filecontent = self.expected_dat_filecontent[:-4]
        expected_dat_filecontent.insert(self.idx_fm1, self.line_adj_phase)
        expected_dat_filecontent.insert(self.idx_fm1 + 1, self.line_adj_ampl)
        expected_dat_filecontent.insert(self.idx_fm2 + 2, self.line_adj_phase)
        expected_dat_filecontent.insert(self.idx_fm2 + 3, self.line_adj_ampl)

        actual_dat_filecontent = load_dat_file(
            self.dat_path, instructions_to_insert=instructions
        )
        assert expected_dat_filecontent == actual_dat_filecontent[:15], (
            f"Expected:\n{pformat(expected_dat_filecontent, width=120)}\nbut "
            f"returned:\n{pformat(actual_dat_filecontent[:15], width=120)}"
        )
