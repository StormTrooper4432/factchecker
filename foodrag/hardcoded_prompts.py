from typing import Dict, List, Optional

HARD_CODED_CLAIMS: Dict[str, Dict[str, object]] = {
    "You must consume protein within 30 minutes of finishing your workout, or you will lose your gains and miss the 'anabolic window' for muscle growth.": {
        "verdict": "false",
        "reasoning": (
            "While post-workout nutrition is important, the 30-minute window is a massive exaggeration. "
            "Total daily protein intake is far more important than a single shake immediately after training."
        ),
        "evidence": [
            "Research shows that the most important factor for muscle growth is total protein intake throughout the day, rather than the specific timing of a single shake.",
            "The body remains sensitized to protein for at least 24 hours following a resistance training session.",
            "If you ate a meal containing protein 1–3 hours before your workout, those amino acids are still in your bloodstream after you finish lifting, making the emergency post-workout shake unnecessary.",
            "Source: https://jissn.biomedcentral.com/articles/10.1186/1550-2783-10-5",
            "Source: https://www.healthline.com/health/fitness-exercise/anabolic-window",
        ],
    },
    "Lifting heavy weights with low reps is only for building bulk, while lifting light weights with high reps is the only way to get 'toned' and defined.": {
        "verdict": "false",
        "reasoning": (
            "Toning is a myth; muscle definition is a result of muscle size and low body fat. "
            "You can build muscle across a wide range of rep schemes, and definition comes from body composition rather than rep count alone."
        ),
        "evidence": [
            "You can build muscle in a wide variety of rep ranges (from 5 up to 30 reps) as long as you are training close to failure.",
            "Definition comes from losing body fat while maintaining muscle mass. High reps do not shape the muscle differently; they just burn slightly more calories during the set.",
            "While you can build muscle with high reps, building maximal strength requires lifting heavier loads (1–5 rep range) to train the nervous system.",
            "Source: https://www.healthline.com/health/exercise-fitness/what-are-reps",
            "Source: https://www.abom.org/wp-content/uploads/2018/12/Quantity_and_Quality_of_Exercise_for_Developing.26-002.pdf",
        ],
    },
    "Natural herbal testosterone boosters (like Tribulus or Fenugreek) will significantly increase your muscle mass and strength similarly to hormonal therapy.": {
        "verdict": "false",
        "reasoning": (
            "Most over-the-counter T-boosters have little to no impact on muscle protein synthesis in healthy men. "
            "Small changes in testosterone are usually within the normal physiological range, which is not enough to cause significant muscle growth."
        ),
        "evidence": [
            "Many herbs marketed as T-boosters may increase libido, but clinical studies show they rarely increase actual serum testosterone levels in healthy individuals.",
            "Even if a supplement increases testosterone slightly, it is usually within the normal physiological range, which is not enough to cause significant muscle growth.",
            "Focusing on sleep, zinc/vitamin D levels (if deficient), and heavy compound lifts is more effective for maintaining healthy testosterone than these supplements.",
            "Source: https://ods.od.nih.gov/factsheets/ExerciseAndAthleticPerformance-HealthProfessional/",
            "Source: https://www.healthline.com/nutrition/best-testosterone-booster-supplements",
        ],
    },
    "Doing cardio on an empty stomach in the morning burns significantly more fat and preserves more muscle than doing cardio after eating.": {
        "verdict": "false",
        "reasoning": (
            "Fasted cardio does not lead to greater fat loss over time compared to fed cardio. "
            "Total fat loss depends on daily caloric deficit, not whether the session occurred before or after eating."
        ),
        "evidence": [
            "While you may burn a higher percentage of fat for fuel during a fasted session, your body compensates by burning more carbohydrates later in the day.",
            "Many people find their intensity drops when training fasted, meaning they burn fewer total calories than they would have if they had eaten a small meal.",
            "High-intensity fasted cardio can actually increase the rate of muscle protein breakdown, which is counterproductive for strength goals.",
            "Source: Journal of the International Society of Sports Nutrition - Body composition changes associated with fasted vs. non-fasted aerobic exercise",
            "Source: Cleveland Clinic - Should You Work Out on an Empty Stomach?",
        ],
    },
    "You should always perform long, static stretches (holding a stretch for 30+ seconds) before lifting to prevent injury and increase strength.": {
        "verdict": "false",
        "reasoning": (
            "Static stretching before lifting can decrease power output and does not reliably prevent injury. "
            "Dynamic stretching is a better warm-up strategy for preparing muscles and joints for heavy lifting."
        ),
        "evidence": [
            "Holding static stretches before a workout relaxes muscles and tendons, which can temporarily reduce their ability to generate explosive force.",
            "Research suggests that dynamic stretching is far superior for preparing the body for heavy lifting.",
            "Static stretching does not reduce the risk of overuse injuries or acute tears during a workout. It is better utilized after a session to improve long-term flexibility.",
            "Source: Mayo Clinic - Stretching: Focus on flexibility",
            "Source: The Journal of Strength & Conditioning Research - Chronic Effect of Static Stretching on Strength",
        ],
    },
}


def get_available_claims() -> List[str]:
    return list(HARD_CODED_CLAIMS.keys())


def get_verdict_for_claim(claim: str) -> Optional[Dict[str, object]]:
    normalized_claim = claim.strip().lower()
    for known_claim, result in HARD_CODED_CLAIMS.items():
        if normalized_claim == known_claim.lower():
            return {"claim": known_claim, **result}
    return None
