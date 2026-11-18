"""Build the multi-turn evaluation set (alignment_plan CONV-03): 100 hand-written chats over the squad_dev_unans corpus.

    python -m evaluation.prepare_conversations          # writes eval_data/conv_dev_conversations.json

Why hand-written: INF-06 let the generator model write its own follow-ups, so a blind spot shared by the writer and the
rewriter could flatter the result. Every chat below was written by a person (with Claude Code, 2026-11-17), not by the
model under test. The last user message of each chat is the one that is measured; the earlier turns are its history.

Five kinds of chat (`category`):
  pronoun        (35) the last question names its subject only by a pronoun ("What does it protect against?")
  topic_switch   (20) the last question is self-contained and about something new: rewriting must NOT pull in the old topic
  vague          (15) a format request with no subject at all ("Tell me more about that.", "Give me the full list.")
  refusal_mid    (15) the turn before the last one was an "I don't know"; the last question points back past it
  unanswerable   (15) a pronoun follow-up whose answer was removed from the corpus: the right reply is "I don't know"

Every chat keeps `standalone`: the same question written out in full (for SQuAD-based ones the original SQuAD question), the
upper bound a perfect rewriter would reach. The gold passage comes from the SQuAD item, so the measures are judge-free:
is the gold passage retrieved, and is the reply a refusal exactly when it should be.

Corpus: squad_dev_unans (550 passages; about half of the dev questions have their answer passage removed). Dev data:
strategies may be compared and tuned on it (CONV-04); a fresh set is needed before any final claim.
"""
import json
import os

from answers import ABSTAIN_MESSAGE
from evaluation.datasets import EVAL_DATA_DIR, load_dataset

CORPUS = "squad_dev_unans"
OUT = os.path.join(EVAL_DATA_DIR, "conv_dev_conversations.json")

TESLA = ("Who was Nikola Tesla?", "A Serbian-American inventor and electrical engineer.")
LUTHER = ("Who was Martin Luther?", "A German theologian who started the Protestant Reformation.")

# (dataset index, earlier question, earlier answer, last question). The standalone version is the SQuAD question itself.
PRONOUN = [
    (0, "What is the immune system?", "A system of biological structures and processes that protects an organism against disease.",
     "What does it protect against?"),
    (74, *TESLA, "How old was he when he became a US citizen?"),
    (153, "Where was Tesla born?", "In Smiljan.", "What disease did he catch after going back there in 1873?"),
    (232, "What were Tesla's views on physics like?", "He held several unconventional views.",
     "Which subatomic particle did he deny existed?"),
    (284, "Who was Tesla's father?", "Milutin Tesla, an Orthodox priest.", "What did he originally want his son to do?"),
    (44, *LUTHER, "Where did he preach his last sermon?"),
    (315, "Where was Martin Luther born?", "In Eisleben.", "And when did he die?"),
    (122, "Where did Luther study?", "At the University of Erfurt.", "How early did he have to get up every day there?"),
    (237, "Which university did Luther enter in 1501?", "The University of Erfurt.",
     "Which subject did he start studying there and then drop?"),
    (92, "Who was Temüjin?", "The birth name of Genghis Khan.", "Who were his three full brothers?"),
    (57, "What was Genghis Khan's name as a young man?", "Temüjin.", "Who captured him and held him prisoner around 1177?"),
    (28, "Who founded the Mongol Empire?", "Genghis Khan.", "Which trading route did he bring a stable political climate to?"),
    (99, "What is the capital of Poland?", "Warsaw.", "Which river runs through it?"),
    (110, "Where in Poland is Warsaw?", "In east-central Poland.", "How many geomorphologic formations is it located on?"),
    (137, "What is Warsaw's symbol?", "The mermaid (syrenka).", "When is the oldest armed seal showing it from?"),
    (195, "Was Warsaw the capital of the Polish-Lithuanian Commonwealth?", "Yes, until 1796.", "Whose army liberated it in 1806?"),
    (31, "Which BBC science-fiction series came back in 2005?", "Doctor Who.", "What was the first episode of the new series called?"),
    (80, "Who is the Master in Doctor Who?", "The Doctor's archenemy, a renegade Time Lord.", "Who was the first actor to play him?"),
    (115, "What is Norman art?", "Art made by the Normans, mostly in the 11th and 12th centuries.", "What is its most famous piece?"),
    (112, "Who was Ethelred?", "Ethelred the Unready, a king of England.", "Where did he flee to?"),
    (158, "What are pyrenoids?", "Structures in the chloroplasts of some hornworts and algae.", "What do they store?"),
    (193, "What are chloroplasts?", "Organelles that carry out photosynthesis in plants and algae.", "What are they descended from?"),
    (185, "Do chloroplasts make sugar?", "Yes, they fix carbon dioxide into sugar during photosynthesis.",
     "What do they use to do that?"),
    (175, "What is the county seat of Fresno County?", "Fresno.", "What does its name mean in Spanish?"),
    (187, "Which city stands on the River Tyne in north-east England?", "Newcastle upon Tyne.", "How far is it from the North Sea?"),
    (350, "Tell me about Newcastle upon Tyne.", "It is a city in North East England.",
     "Which network of European cities is it a member of?"),
    (283, "Where was Super Bowl 50 played?", "At Levi's Stadium in Santa Clara, California.", "How much did it cost to build?"),
    (327, "What is Levi's Stadium?", "The home stadium of the San Francisco 49ers.",
     "Who slipped on its turf in week 6 of the 2015 season?"),
    (100, "Does the V&A have a library?", "Yes, the National Art Library.", "What other name is it known by?"),
    (141, "What are the Cast Courts?", "Galleries of the V&A that hold plaster casts of famous sculptures.",
     "Which ancient Roman monument is replicated there in full scale?"),
    (241, "Who bought ABC in 1985?", "Capital Cities Communications.", "How much did they pay for it?"),
    (296, "What was NYPD Blue?", "An ABC police drama series.", "How many seasons did it last?"),
    (106, "What is singlet oxygen?", "A name for several higher-energy forms of molecular oxygen.", "What is it more reactive towards?"),
    (167, "What are natural killer cells?", "A part of the innate immune system.",
     "By what condition do they recognise the cells they should target?"),
    (365, "What is Shark Tank?", "An American reality show where entrepreneurs pitch to investors.", "Which show was it based on?"),
]

# (dataset index, earlier question, earlier answer). The last question is the SQuAD question, unchanged and self-contained.
TOPIC_SWITCH = [
    (14, *TESLA),
    (15, "What river runs through Warsaw?", "The Vistula River."),
    (27, "What is the immune system?", "A system that protects an organism against disease."),
    (35, *LUTHER),
    (56, "Who founded the Mongol Empire?", "Genghis Khan."),
    (63, "What are chloroplasts?", "Organelles that carry out photosynthesis."),
    (69, "What is Norman art?", "Art made by the Normans in the 11th and 12th centuries."),
    (76, "What does the name Fresno mean?", "Ash tree."),
    (81, "What is Doctor Who?", "A British science-fiction television series."),
    (97, "How much did Levi's Stadium cost?", "About $1.2 billion."),
    (120, "What is the V&A?", "A museum of art and design in London."),
    (129, "Who bought ABC in 1985?", "Capital Cities Communications."),
    (135, "What are pyrenoids?", "Structures in the chloroplasts of some algae."),
    (154, "What is Shark Tank?", "A reality TV show about investors and entrepreneurs."),
    (164, "What is singlet oxygen?", "A higher-energy form of molecular oxygen."),
    (166, "Where is Newcastle upon Tyne?", "In North East England."),
    (184, "What was NYPD Blue?", "An ABC police drama."),
    (214, "Who was Temüjin?", "Genghis Khan's birth name."),
    (215, "What are natural killer cells?", "Part of the innate immune system."),
    (268, "Who was Ethelred?", "A king of England."),
]

# (dataset index, earlier answer, last question, standalone version). The earlier question is the SQuAD question.
VAGUE = [
    (70, '"Bricks for Warsaw".', "Tell me more about that.", "Tell me more about the Bricks for Warsaw campaign after World War II."),
    (101, "11.", "What happened to him exactly? Give me the details.",
     "What happened when Thomas Davis broke his arm in the NFC Championship game?"),
    (113, "Over $40 million.", "Can you explain that in more detail?",
     "What did the Super Bowl 50 Host Committee raise over $40 million for?"),
    (119, "Chicago's physics department.", "Give me a detailed list of what else they contributed to.",
     "What major developments have University of Chicago scholars contributed to?"),
    (133, "Black and yellow.", "Summarise that whole campaign for me.",
     "Summarise ABC's web-based promotional campaign and its 2001 logo."),
    (144, "The Guanabara Confession of Faith.", "Tell me the full story.", "What happened to the first Huguenots who went to Brazil?"),
    (145, "Dublin, Cork, Youghal and Waterford.", "List everything else the text says about them in Ireland.",
     "What did the Huguenots do in Ireland?"),
    (161, "250,000 feet.", "Describe the facility in more detail.",
     "Describe the Launch Operations Center facilities, including its vacuum chambers."),
    (176, "Transportation, sewer, hazardous waste and water.", "Can you elaborate?",
     "How does Engineering News-Record rank construction firms?"),
    (178, "The Common Core.", "Explain more about it.", "Explain the University of Chicago's Common Core liberal-arts curriculum."),
    (189, "Otter, beaver and hundreds of bird species.", "What else? Give me the full list.",
     "What natural reserves and wildlife are there in Warsaw?"),
    (204, "Some Asian, African and Caribbean countries.", "Explain in detail.",
     "Where is official corporal punishment still commonplace, and how is it carried out?"),
    (207, "The public.", "Give me more details on when it sits.",
     "When does the Scottish Parliament sit, and who can attend its debates and meetings?"),
    (227, "Mechanical oscillators and generators, electrical discharge tubes and early X-ray imaging.", "Tell me more.",
     "What experiments did Tesla carry out?"),
    (245, "8 to 15 per cell.", "Can you explain that more fully?",
     "How many chloroplasts are in stomatal guard cells compared with other plant cells?"),
]

# (dataset index, earlier question, earlier answer, index of an UNANSWERABLE middle question, last question)
REFUSAL_MID = [
    (24, *LUTHER, 7, "When is he commemorated by the Church of England?"),
    (29, "Who was Jochi?", "Genghis Khan's eldest son.", 10, "Where did he stay after his father sent for his sons in spring 1223?"),
    (30, "What was the NIF in Sudan?", "The National Islamic Front, an Islamist political party.", 17,
     "What did it try to unify Islamist opposition against?"),
    (50, "What was the Il-khanate?", "A Mongol khanate based in Persia.", 26, "When did it experiment with paper money?"),
    (125, "Who was William Hartnell?", "The actor who played the First Doctor in Doctor Who.", 38, "Which episode featured his return?"),
    (136, "Who is the Doctor?", "The main character of Doctor Who, a time-travelling alien.", 47,
     "What kind of beings does he usually take with him on his travels?"),
    (148, "What did Luther write about faith?", "That people are saved by faith alone.", 49,
     "What did he explain about acquiring God's grace?"),
    (160, "Did Tesla and Edison share a Nobel Prize?", "No, although a 1915 news report said they would.", 58,
     "What was the rumoured reason they were not awarded it?"),
    (163, "What is vitamin D?", "A vitamin the body needs for bones and the immune system.", 61, "What is its active form called?"),
    (197, "Did Tesla start his own company?", "Yes, Tesla Electric Light & Manufacturing.", 62, "What did that company do?"),
    (201, "What was Anglo-Norman?", "The language the Normans spoke in England.", 13, "What was its final form?"),
    (269, "Did Tesla talk about resonant frequencies?", "Yes, in several interviews.", 19,
     "What did he claim they could do in the World Today article?"),
    (287, "Which companies advertised during Super Bowl 50?", "Many, including car makers and game companies.", 40,
     "Which video game company showed its first ad there?"),
    (294, "Who headlined the Super Bowl 50 halftime show?", "Coldplay, with Beyoncé and Bruno Mars.", 64,
     "Which single had Beyoncé and the band collaborated on?"),
    (333, "What are ctenophores?", "Comb jellies, a group of marine animals.", 67, "What does their internal cavity contain?"),
]

# (dataset index of an UNANSWERABLE question, earlier question, earlier answer, last question)
UNANSWERABLE = [
    (3, *LUTHER, "How many ways did he justify his opposition to revolt?"),
    (16, "Who was Martin Luther?", "A German monk and theologian.", "When did the Pope excommunicate him?"),
    (22, "What did Luther write about other religions?", "He wrote about Judaism and Islam, among others.",
     "How did he view the Ottoman Turks?"),
    (20, "Who was the Panthers' quarterback in Super Bowl 50?", "Cam Newton.", "Who stripped the ball from him on a 3rd and nine?"),
    (46, "Which team represented the NFC in Super Bowl 50?", "The Carolina Panthers.", "Who did they beat to become NFC champions?"),
    (47, "What is the Amazon rainforest?", "A large tropical rainforest in South America.", "How many species of trees grow in it?"),
    (59, "What are chloroplasts?", "Organelles that carry out photosynthesis.", "What does their peripheral reticulum do?"),
    (37, "What is DNA?", "The molecule that carries genetic information.", "What makes it vulnerable to deamination?"),
    (38, "What is the Sahara?", "The largest hot desert in the world.", "How much dust is blown out of it each year?"),
    (7, "What is Lake Constance?", "A lake on the Rhine at the northern foot of the Alps.",
     "Besides Germany and Switzerland, where else is it?"),
    (49, "What is the capital of Victoria in Australia?", "Melbourne.", "Which political party is strongest in its working-class suburbs?"),
    (40, "Does the V&A have a ceramics collection?", "Yes, a large ceramics and glass collection.",
     "Its largest objects were produced in which countries?"),
    (26, "What was the Democratic Republic of Afghanistan?", "The communist state in Afghanistan from 1978.", "When did it collapse?"),
    (52, "Who was William Shirley?", "A governor of the Province of Massachusetts Bay.", "How did he feel about French advancement?"),
    (23, "Who was Tanaghrisson?", "A Native American leader allied with the British.", "Why was he supporting British efforts?"),
]


def chat(n, category, item, history, question, standalone, expect):
    return {"id": f"c{n:03d}", "category": category, "history": history, "question": question, "standalone": standalone,
            "expect": expect, "source_id": item["id"], "ground_truth": item["ground_truth"],
            "gold_passages": item["gold_passages"]}


def turn(q, a):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def build():
    corpus_path, items = load_dataset(CORPUS)
    with open(corpus_path, encoding="utf-8") as f:
        corpus = {line.strip() for line in f if line.strip()}

    def answerable(i):
        it = items[i]
        present = all(p.strip() in corpus for p in it["gold_passages"])
        if bool(it["answerable"]) != present:
            raise SystemExit(f"Item {i} says answerable={it['answerable']} but its passage is {'in' if present else 'not in'} the corpus")
        return it["answerable"]

    out, used = [], set()
    def add(category, i, history, question, standalone, expect):
        need = expect == "answer"
        if answerable(i) != need:
            raise SystemExit(f"{category}: item {i} answerable={items[i]['answerable']} but the chat expects '{expect}'")
        if (category, i) in used:
            raise SystemExit(f"{category}: item {i} used twice")
        used.add((category, i))
        out.append(chat(len(out) + 1, category, items[i], history, question, standalone, expect))

    for i, q, a, last in PRONOUN:
        add("pronoun", i, turn(q, a), last, items[i]["question"].strip(), "answer")
    for i, q, a in TOPIC_SWITCH:
        add("topic_switch", i, turn(q, a), items[i]["question"].strip(), items[i]["question"].strip(), "answer")
    for i, a, last, standalone in VAGUE:
        add("vague", i, turn(items[i]["question"].strip(), a), last, standalone, "answer")
    for i, q, a, mid, last in REFUSAL_MID:
        if answerable(mid):
            raise SystemExit(f"refusal_mid: middle question {mid} is answerable")
        history = turn(q, a) + turn(items[mid]["question"].strip(), ABSTAIN_MESSAGE)
        add("refusal_mid", i, history, last, items[i]["question"].strip(), "answer")
    for i, q, a, last in UNANSWERABLE:
        add("unanswerable", i, turn(q, a), last, items[i]["question"].strip(), "refuse")

    meta = {"name": "conv_dev", "corpus": CORPUS, "n": len(out),
            "categories": {c: sum(r["category"] == c for r in out) for c in
                           ("pronoun", "topic_switch", "vague", "refusal_mid", "unanswerable")},
            "note": "Hand-written 2026-11-17 (see evaluation/prepare_conversations.py). Dev data: compare and tune here, "
                    "confirm on a fresh set before claiming."}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "conversations": out}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} chats to {OUT}: {meta['categories']}")
    return out


if __name__ == "__main__":
    build()
