"""Phase 1 tests: status map integrity and parser behaviour on a synthetic target.

Run:  .venv/bin/python -m pytest -q
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import yaml
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

parse = __import__("02_parse_tt_xml")  # numeric module name: import by string


# --------------------------------------------------------------------------- status map

XSD_STATUS = [
    "selected", "cloned", "expression tested", "expressed", "biological assay",
    "biophysical analysis", "soluble", "membrane protein solubilized", "purified",
    "mass spec verified", "crystallized", "diffraction-quality crystals", "diffraction",
    "native diffraction-data", "phasing diffraction-data", "crystal structure",
    "HSQC satisfactory", "NMR assigned", "NMR backbone resonances assigned",
    "NMR sidechain resonances assigned", "NMR structure", "in BMRB", "EM images",
    "EM reconstruction", "EM reconstruction in EMDB", "EM fitted model", "in PDB",
    "work stopped", "test target", "other",
]


@pytest.fixture(scope="module")
def smap():
    return yaml.safe_load((ROOT / "config" / "status_map.yaml").read_text())


def test_every_xsd_status_is_mapped(smap):
    missing = [s for s in XSD_STATUS if s not in smap["statuses"]]
    assert not missing, missing


def test_ladder_is_0_to_8(smap):
    assert list(smap["ladder"]) == list(range(9))
    names = set(smap["ladder"].values())
    for raw, m in smap["statuses"].items():
        if m["stage_ord"] is not None:
            assert 0 <= m["stage_ord"] <= 8, raw
            assert m["canon"] in names, raw
            assert smap["ladder"][m["stage_ord"]] == m["canon"], raw


def test_non_ladder_values_have_null_stage(smap):
    for raw in ("work stopped", "other", "test target", "biological assay"):
        assert smap["statuses"][raw]["stage_ord"] is None


def test_work_stopped_is_terminal_not_a_stage(smap):
    assert smap["statuses"]["work stopped"]["terminal"] is True


# --------------------------------------------------------------------------- parser

SAMPLE = """<?xml version="1.0" encoding="ISO-8859-1"?>
<targetTrack>
<formatVersion>1.4.1</formatVersion><site>TargetTrack</site>
<contactInfoList><contactInfo id="X-1"><contactInfoId>1</contactInfoId><name>n</name></contactInfo></contactInfoList>
<protocolList>
  <protocol id="JCSG-P1"><protocolId>P1</protocolId><protocolName>Expr</protocolName>
    <protocolType>expression</protocolType><protocolText>BL21(DE3) 37C</protocolText></protocol>
</protocolList>
<target id="JCSG-419184">
  <targetId>419184</targetId><dateCreated>2011-01-05</dateCreated><dateUpdated>2012-06-01</dateUpdated>
  <laboratoryList><lab>JCSG</lab></laboratoryList>
  <contactInfoRefList><contactInfoId>1</contactInfoId></contactInfoRefList>
  <projectList><project><projectName>PSI</projectName><projectId>2</projectId></project></projectList>
  <targetRationale>r</targetRationale>
  <targetCategoryList><targetCategory><targetCategoryName>biomedical</targetCategoryName></targetCategory></targetCategoryList>
  <targetProteinType>single-domain protein</targetProteinType>
  <status>in PDB</status>
  <targetSequenceList>
    <targetSequence id="1">
      <oneLetterCode>
        MKT AYIA
        KQR
      </oneLetterCode>
      <sequenceType>protein</sequenceType><sequenceChemicalType>protein</sequenceChemicalType>
      <sequenceConstructType>full length ORF</sequenceConstructType>
      <sourceOrganism><scientificName>Parabacteroides merdae</scientificName><taxDB>NCBI</taxDB><taxId>411477</taxId></sourceOrganism>
      <sequenceName>thua-like</sequenceName>
      <databaseRefList><databaseRef><databaseName>PDB</databaseName><databaseId>4e5v</databaseId></databaseRef></databaseRefList>
    </targetSequence>
    <targetSequence id="2">
      <oneLetterCode>ACGT</oneLetterCode>
      <sequenceType>predicted dna</sequenceType><sequenceChemicalType>dna</sequenceChemicalType>
      <sequenceConstructType>full length ORF</sequenceConstructType>
      <sourceOrganism><scientificName>x</scientificName><taxDB></taxDB><taxId></taxId></sourceOrganism>
      <sequenceName>gene</sequenceName>
    </targetSequence>
  </targetSequenceList>
  <databaseRefList><databaseRef><databaseName>UniProt</databaseName><databaseId>A7V1Y5</databaseId></databaseRef></databaseRefList>
  <trialList>
    <trial id="1">
      <dateUpdated>2012-06-01</dateUpdated>
      <contactInfoRefList><contactInfoId>1</contactInfoId></contactInfoRefList>
      <status>in PDB</status>
      <statusHistoryList>
        <statusHistory id="1"><lab>JCSG</lab><status>selected</status><dateComplete>2011-01-05</dateComplete></statusHistory>
        <statusHistory id="2"><lab>JCSG</lab><status>cloned</status><dateComplete>2011-02-01</dateComplete><stepDurationDays>27</stepDurationDays></statusHistory>
        <statusHistory id="3"><lab>JCSG</lab><status>in PDB</status><dateComplete>2012-04-04</dateComplete></statusHistory>
      </statusHistoryList>
      <stopDetails><stopStatus>structure successful</stopStatus></stopDetails>
      <trialSequenceList><trialSequence id="1"><oneLetterCode>GSMKTAYIAKQR</oneLetterCode>
        <sequenceConstructType>tagged protein</sequenceConstructType>
        <sequenceDetails>N-terminal His tag</sequenceDetails></trialSequence></trialSequenceList>
      <trialProtocolList><protocolRef id="1"><protocolId>P1</protocolId><protocolType>expression</protocolType></protocolRef></trialProtocolList>
    </trial>
  </trialList>
</target>
</targetTrack>
"""


@pytest.fixture(scope="module")
def parsed():
    known = {"JCSG", "MCSG"}
    out = {}
    for _, el in etree.iterparse(io.BytesIO(SAMPLE.encode("latin-1")), events=("end",), tag=("target", "protocol")):
        if el.tag == "protocol":
            out["protocol"] = parse.extract_protocol(el, known)
        else:
            out["target"], out["seqs"], out["hist"], out["trials"] = parse.extract_target(el, known, "JCSG.xml.gz")
    return out


def test_centre_from_id_prefix(parsed):
    assert parsed["target"]["centre"] == "JCSG"
    assert parsed["target"]["target_id"] == "JCSG-419184"
    assert parsed["target"]["local_id"] == "419184"


def test_sequence_whitespace_stripped_and_hashed(parsed):
    t = parsed["target"]
    assert t["sequence"] == "MKTAYIAKQR"
    assert t["seq_len"] == 10
    assert t["seq_md5"] == parse.md5("MKTAYIAKQR")


def test_primary_is_longest_protein_not_dna(parsed):
    seqs = parsed["seqs"]
    assert len(seqs) == 2
    prim = [s for s in seqs if s["is_primary"]]
    assert len(prim) == 1 and prim[0]["chem_type"] == "protein"
    assert parsed["target"]["organism"] == "Parabacteroides merdae"
    assert parsed["target"]["taxon_id"] == "411477"


def test_pdb_ids_collected_upper_from_sequence_refs(parsed):
    assert parsed["target"]["pdb_ids"] == "4E5V"
    assert "UniProt:A7V1Y5" in parsed["target"]["database_refs"]


def test_status_history_rows(parsed):
    h = parsed["hist"]
    assert [r["status_raw"] for r in h] == ["selected", "cloned", "in PDB"]
    assert h[1]["step_duration_days"] == 27
    assert h[0]["step_duration_days"] is None
    assert all(r["trial_id"] == 1 for r in h)


def test_first_last_seen_span_all_dates(parsed):
    assert parsed["target"]["first_seen"] == "2011-01-05"
    assert parsed["target"]["last_seen"] == "2012-06-01"


def test_trial_row(parsed):
    tr = parsed["trials"][0]
    assert tr["stop_status"] == "structure successful"
    assert tr["protocol_refs"] == "JCSG-P1"
    assert tr["protocol_types"] == "expression"
    assert tr["sequence"] == "GSMKTAYIAKQR"
    assert tr["construct_type"] == "tagged protein"
    assert "N-terminal His tag" in tr["free_text_notes"]


def test_protocol_row(parsed):
    p = parsed["protocol"]
    assert p["protocol_id"] == "JCSG-P1" and p["centre"] == "JCSG"
    assert p["protocol_type"] == "expression" and "BL21" in p["text"]


def test_schemas_accept_rows(parsed):
    import pyarrow as pa
    pa.Table.from_pylist([parsed["target"]], schema=parse.SCHEMAS["targets"])
    pa.Table.from_pylist(parsed["seqs"], schema=parse.SCHEMAS["target_sequences"])
    pa.Table.from_pylist(parsed["hist"], schema=parse.SCHEMAS["status_history"])
    pa.Table.from_pylist(parsed["trials"], schema=parse.SCHEMAS["trials"])
    pa.Table.from_pylist([parsed["protocol"]], schema=parse.SCHEMAS["protocols"])
