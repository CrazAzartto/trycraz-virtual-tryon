"""Tests for the recommendation module (brief section 10.2).

Uses tests/fixtures/catalogue_fixture.csv: 18 hand-annotated garments (metadata only). The
scenario assertions encode conventional dress-code expectations, written BEFORE looking at the
engine's output. They check behaviour (what ranks above what), not exact scores.
"""

from pathlib import Path

import pytest
from PIL import Image

from recommendation.catalogue import CatalogueError, garment_from_type, load_catalogue
from recommendation.context_processor import Context, ContextError, build_profile
from recommendation.garment_analyzer import analyse_garment, dominant_colours
from recommendation.knowledge import default_knowledge, load_knowledge
from recommendation.recommendation_engine import RecommendationEngine, normalise_weights
from recommendation.scoring import formality_fit, type_phrase, weather_fit

FIXTURE = Path(__file__).parent / "fixtures" / "catalogue_fixture.csv"


@pytest.fixture(scope="module")
def garments():
    return load_catalogue(FIXTURE, require_images=False)


@pytest.fixture(scope="module")
def engine(garments):
    return RecommendationEngine(garments)


def top_ids(engine, k=3, **ctx):
    return [r.garment.id for r in engine.recommend(Context(**ctx), top_k=k).items]


def rank_of(result, garment_id):
    return next(r.rank for r in result.all_ranked if r.garment.id == garment_id)


# ---------------------------------------------------------------------------
# Knowledge base and catalogue
# ---------------------------------------------------------------------------
def test_knowledge_base_loads_and_weights_sum_to_one():
    kb = load_knowledge()
    assert abs(sum(kb.weights.values()) - 1) < 1e-9
    assert {"interview", "wedding", "sports_outdoor"} <= set(kb.occasions)


def test_fixture_catalogue_loads(garments):
    assert len(garments) == 18 and {g.tryon_type for g in garments} == {"upper", "lower", "overall"}


def test_catalogue_validation_reports_row_numbers(tmp_path):
    bad = tmp_path / "bad.csv"
    text = FIXTURE.read_text().splitlines()
    text[1] = text[1].replace("shirt_formal", "spacesuit").replace(",4,2,", ",9,2,")
    bad.write_text("\n".join(text))
    with pytest.raises(CatalogueError) as err:
        load_catalogue(bad, require_images=False)
    joined = " ".join(err.value.problems)
    assert "row 2" in joined and "spacesuit" in joined


def test_context_validation():
    with pytest.raises(ContextError):
        build_profile(Context(occasion="moon_landing"))
    with pytest.raises(ContextError):
        build_profile(Context(occasion="office", colours=("ultraviolet",)))


# ---------------------------------------------------------------------------
# Factor functions
# ---------------------------------------------------------------------------
def test_formality_factor_is_monotonic():
    profile = build_profile(Context(occasion="interview"))
    values = [formality_fit(garment_from_type(t), profile).value for t in ("tank_top", "t_shirt", "shirt_casual", "shirt_formal")]
    assert values == sorted(values)


def test_indoor_softens_weather():
    puffer = garment_from_type("puffer_jacket")
    outdoor = weather_fit(puffer, build_profile(Context(occasion="travel", setting="outdoor", weather="cold"))).value
    indoor = weather_fit(puffer, build_profile(Context(occasion="travel", setting="indoor", weather="cold"))).value
    assert outdoor > indoor


def test_type_phrase_grammar():
    profile = build_profile(Context(occasion="office"))
    assert type_phrase(profile, "jeans") == "jeans are"
    assert type_phrase(profile, "hoodie") == "a hoodie is"


# ---------------------------------------------------------------------------
# Scenarios (brief examples + more)
# ---------------------------------------------------------------------------
def test_interview_formal_indoor(engine):
    top = top_ids(engine, occasion="interview", setting="indoor", weather="mild", styles=("formal",))
    assert set(top) <= {"f01", "f02", "f03", "f14"}          # formal shirts, blazer, formal trousers


def test_interview_excludes_hoodie_and_shorts(engine):
    result = engine.recommend(Context(occasion="interview"))
    excluded = {g.id for g, _ in result.excluded}
    assert {"f07", "f11", "f15"} <= excluded                  # hoodie, tank top, shorts
    assert all(r.garment.id not in excluded for r in result.all_ranked)


def test_casual_hot_outdoor(engine):
    top = top_ids(engine, occasion="casual_outing", setting="outdoor", weather="hot")
    assert set(top) & {"f04", "f05", "f15", "f16"}           # t-shirts, shorts, summer dress
    assert not set(top) & {"f03", "f10", "f07"}               # no blazer, puffer, hoodie


def test_outdoor_rain_prefers_water_resistant(engine):
    result = engine.recommend(Context(occasion="travel", setting="outdoor", weather="rainy", categories=("upper",)))
    assert result.items[0].garment.water_resistant


def test_cold_outdoor_prefers_warm(engine):
    result = engine.recommend(Context(occasion="casual_outing", setting="outdoor", weather="cold", categories=("upper",)))
    assert result.items[0].garment.warmth >= 3.5


def test_wedding_puts_kurta_or_blazer_first(engine):
    assert top_ids(engine, k=1, occasion="wedding", categories=("upper",))[0] in {"f12", "f03"}


def test_sports_prefers_athletic(engine):
    top = top_ids(engine, occasion="sports_outdoor", setting="outdoor", weather="hot")
    assert "f11" in top and "f03" not in top


def test_ranking_changes_with_context(engine):
    interview = engine.recommend(Context(occasion="interview", categories=("upper",)))
    casual = engine.recommend(Context(occasion="casual_outing", weather="hot", setting="outdoor", categories=("upper",)))
    assert rank_of(interview, "f03") < rank_of(interview, "f05")
    assert rank_of(casual, "f05") < rank_of(casual, "f03")


def test_colour_preference_is_respected(engine):
    base = engine.recommend(Context(occasion="casual_outing", categories=("upper",)))
    pref = engine.recommend(Context(occasion="casual_outing", categories=("upper",), colours=("red",)))
    assert rank_of(pref, "f06") < rank_of(base, "f06") or rank_of(base, "f06") == 1


def test_formality_shift_moves_ranking(engine):
    casual = engine.recommend(Context(occasion="party", formality_shift=-1.0, categories=("upper",)))
    formal = engine.recommend(Context(occasion="party", formality_shift=+1.0, categories=("upper",)))
    assert rank_of(formal, "f03") < rank_of(casual, "f03")


def test_category_and_wear_line_filters(engine):
    lower = engine.recommend(Context(occasion="office", categories=("lower",)))
    assert {r.garment.tryon_type for r in lower.all_ranked} == {"lower"}
    mens = engine.recommend(Context(occasion="party", wear_line="menswear"))
    assert all(r.garment.wear_line in ("menswear", "unisex") for r in mens.all_ranked)


def test_ranking_is_deterministic(engine):
    ctx = Context(occasion="college", weather="mild")
    assert [r.garment.id for r in engine.recommend(ctx).all_ranked] == [r.garment.id for r in engine.recommend(ctx).all_ranked]


# ---------------------------------------------------------------------------
# Explanations must agree with the scores
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("occasion", sorted(default_knowledge().occasions))
@pytest.mark.parametrize("weather", ["hot", "cold", "rainy"])
def test_explanations_consistent_with_factors(engine, occasion, weather):
    kb = default_knowledge()
    result = engine.recommend(Context(occasion=occasion, weather=weather, setting="outdoor", colours=("navy",)))
    for rec in result.all_ranked:
        for factor in rec.explanation.reason_factors:
            assert rec.factors[factor].value >= kb.strong
        for factor in rec.explanation.caution_factors:
            assert rec.factors[factor].value <= kb.weak
        assert abs(rec.score - sum(result.weights[k] * f.value for k, f in rec.factors.items())) < 1e-9
        assert rec.explanation.summary


NEGATIVE_WORDS = ("not ", "too ", "much more", "fallback", "bolder")


@pytest.mark.parametrize("occasion", sorted(default_knowledge().occasions))
@pytest.mark.parametrize("weather", ["hot", "mild", "cold", "rainy"])
def test_reasons_never_use_negative_wording(engine, occasion, weather):
    """Regression test: a reason once read 'Recommended because ... more casual than the target'."""
    result = engine.recommend(Context(occasion=occasion, weather=weather, setting="outdoor"))
    for rec in result.all_ranked:
        for _, phrase in rec.explanation.reasons:
            assert not any(word in phrase for word in NEGATIVE_WORDS), (rec.garment.name, phrase)


def test_scores_are_bounded(engine):
    for rec in engine.recommend(Context(occasion="office")).all_ranked:
        assert 0.0 <= rec.score <= 1.0


# ---------------------------------------------------------------------------
# CLIP factor plumbing (fake scorer: real CLIP is tested separately on the laptop)
# ---------------------------------------------------------------------------
class FakeClipScorer:
    """Pretends every image looks like interview wear in hot weather."""

    def garment_probs_for(self, garment):
        kb = default_knowledge()
        occ = {k: (0.9 if k == "interview" else 0.1 / (len(kb.occasions) - 1)) for k in kb.occasions}
        wx = {k: (0.7 if k == "hot" else 0.1) for k in kb.weather}
        return {"occasion": occ, "weather": wx}


def test_clip_weight_redistributed_when_unavailable(engine):
    result = engine.recommend(Context(occasion="office"))
    assert not result.clip_used and result.weights["clip"] == 0 and abs(sum(result.weights.values()) - 1) < 1e-9


def test_clip_factor_used_when_available(garments):
    engine = RecommendationEngine(garments, clip_scorer=FakeClipScorer())
    result = engine.recommend(Context(occasion="interview", weather="hot"))
    assert result.clip_used and result.items[0].factors["clip"].value == pytest.approx(1.0)
    other = engine.recommend(Context(occasion="party", weather="cold"))
    assert other.items[0].factors["clip"].value < 0.5


def test_normalise_weights_drop():
    w = normalise_weights({"occasion": 1, "formality": 1, "weather": 0, "colour": 0, "style": 0, "clip": 2}, drop=("clip",))
    assert w == {"occasion": 0.5, "formality": 0.5, "weather": 0, "colour": 0, "style": 0, "clip": 0}


# ---------------------------------------------------------------------------
# Uploaded garment analysis
# ---------------------------------------------------------------------------
def test_dominant_colour_of_navy_shirt_on_white():
    image = Image.new("RGB", (300, 400), (250, 250, 250))
    image.paste((28, 38, 85), (60, 60, 240, 340))
    analysis = dominant_colours(image)
    assert analysis.colours[0][0] == "navy" and 0.3 < analysis.foreground_fraction < 0.6


def test_two_colour_garment():
    image = Image.new("RGB", (300, 400), (250, 250, 250))
    image.paste((200, 30, 40), (60, 60, 240, 200))
    image.paste((40, 140, 60), (60, 200, 240, 340))
    names = [n for n, _ in dominant_colours(image).colours]
    assert set(names[:2]) == {"red", "green"}


def test_assess_uploaded_garment(engine):
    image = Image.new("RGB", (300, 400), (250, 250, 250))
    image.paste((240, 240, 240), (60, 60, 240, 340))
    analysis = analyse_garment(image, garment_type="hoodie")
    assert analysis.garment.estimated and analysis.type_source == "user"
    rec, exclusion = engine.assess(analysis.garment, Context(occasion="interview"))
    assert exclusion and "not appropriate" in exclusion
    rec_ok, exclusion_ok = engine.assess(garment_from_type("blazer"), Context(occasion="interview"))
    assert exclusion_ok is None and rec_ok.score > rec.score
