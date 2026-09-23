from crownai.archive import anonymize, survey


def test_anonymize_hides_names_keeps_technical_words():
    assert anonymize("Иванова_И.И.-2026-08-30-LowerJaw.stl") == "*_*.*.-*-08-30-LowerJaw.stl"
    assert anonymize("Petrov 33 prep.stl") == "* 33 prep.stl"
    assert anonymize("Скан челюсти (37).ply") == "Скан челюсти (37).ply"


def test_survey_counts_case_folders(tmp_path):
    case = tmp_path / "Smith_John_2025"
    case.mkdir()
    (case / "Smith.dentalProject").write_text("<x/>")
    (case / "Smith-UpperJaw.stl").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("-")
    rep = survey(tmp_path)
    assert rep["case_folders"] == 1
    assert rep["files_by_extension"][".stl"] == 1
    names = rep["examples"][0]["names"]
    assert all("Smith" not in n for n in names) and "*-UpperJaw.stl" in names
