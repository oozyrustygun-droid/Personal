const TelegramBot = require("node-telegram-bot-api");
const Anthropic = require("@anthropic-ai/sdk");
const { createClient } = require("@supabase/supabase-js");
const https = require("https");
const http = require("http");
require("dotenv").config();

const bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN, { polling: true });
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_ANON_KEY);

function today() {
  return new Date().toLocaleString("en-CA", { timeZone: "America/New_York" }).split(",")[0];
}

// Extract the first JSON object from a Claude response (handles stray markdown/backticks).
function extractJson(text) {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("No JSON found in response");
  return JSON.parse(match[0]);
}

function formatMeal(meal) {
  return (
    `*${meal.food_name}*\n` +
    `Calories: ${meal.calories} kcal\n` +
    `Protein: ${meal.protein}g | Carbs: ${meal.carbs}g | Fat: ${meal.fat}g\n` +
    `Fiber: ${meal.fiber}g | Sugar: ${meal.sugar}g`
  );
}

function formatTotals(rows) {
  return rows.reduce(
    (acc, m) => {
      acc.calories += m.calories || 0;
      acc.protein += m.protein || 0;
      acc.carbs += m.carbs || 0;
      acc.fat += m.fat || 0;
      acc.fiber += m.fiber || 0;
      acc.sugar += m.sugar || 0;
      return acc;
    },
    { calories: 0, protein: 0, carbs: 0, fat: 0, fiber: 0, sugar: 0 }
  );
}

function parseNutrition(text) {
  const raw = typeof text === "string" ? extractJson(text) : text;
  return {
    food_name: raw.food_name || raw.name || raw.food || raw.item || "Unknown Food",
    calories: Math.round(Number(raw.calories) || 0),
    protein: Math.round(Number(raw.protein) || 0),
    carbs: Math.round(Number(raw.carbs) || raw.carbohydrates || 0),
    fat: Math.round(Number(raw.fat) || raw.fats || 0),
    fiber: Math.round(Number(raw.fiber) || raw.dietary_fiber || 0),
    sugar: Math.round(Number(raw.sugar) || raw.sugars || 0),
    notes: raw.notes || raw.note || raw.assumptions || "",
  };
}

function fetchImageAsBase64(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      const chunks = [];
      res.on("data", (chunk) => chunks.push(chunk));
      res.on("end", () => resolve(Buffer.concat(chunks).toString("base64")));
      res.on("error", reject);
    });
  });
}

function hasMultipleItems(text) {
  return [",", " and ", " with ", "\n", "+", "&"].some((sep) =>
    text.toLowerCase().includes(sep)
  );
}

const SINGLE_PROMPT = `You are a nutrition expert. Respond ONLY with a raw JSON object. No markdown. No backticks. Start with { and end with }.
{
  "food_name": "name of the food",
  "calories": 0,
  "protein": 0,
  "carbs": 0,
  "fat": 0,
  "fiber": 0,
  "sugar": 0,
  "notes": "portion size assumptions"
}
All numbers must be whole integers. All macros in grams. Calories in kcal.`;

const MULTI_PROMPT = `You are a nutrition expert. Break down EACH food item separately then give a total. Respond ONLY with a raw JSON object. No markdown. No backticks. Start with { and end with }.
{
  "items": [
    { "name": "item name", "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0, "sugar": 0 }
  ],
  "total": {
    "food_name": "combined meal description",
    "calories": 0,
    "protein": 0,
    "carbs": 0,
    "fat": 0,
    "fiber": 0,
    "sugar": 0,
    "notes": "portion size assumptions"
  }
}
All numbers must be whole integers. All macros in grams. Calories in kcal.`;

async function analyzeFoodImage(imageUrl) {
  const base64Image = await fetchImageAsBase64(imageUrl);
  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 600,
    messages: [
      {
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: base64Image } },
          { type: "text", text: "Analyze this food image.\n\n" + SINGLE_PROMPT },
        ],
      },
    ],
  });
  return { isBreakdown: false, ...parseNutrition(response.content[0].text) };
}

async function analyzeFoodText(description) {
  const isMultiple = hasMultipleItems(description);
  const response = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 800,
    messages: [
      {
        role: "user",
        content: 'Analyze this food: "' + description + '"\n\n' + (isMultiple ? MULTI_PROMPT : SINGLE_PROMPT),
      },
    ],
  });

  const text = response.content[0].text;
  const raw = extractJson(text);

  if (raw.items && raw.total) {
    return {
      isBreakdown: true,
      items: raw.items.map((item) => ({
        name: item.name || "Unknown",
        calories: Math.round(Number(item.calories) || 0),
        protein: Math.round(Number(item.protein) || 0),
        carbs: Math.round(Number(item.carbs) || 0),
        fat: Math.round(Number(item.fat) || 0),
        fiber: Math.round(Number(item.fiber) || 0),
        sugar: Math.round(Number(item.sugar) || 0),
      })),
      ...parseNutrition(raw.total),
    };
  }

  return { isBreakdown: false, ...parseNutrition(raw) };
}

async function getAISummary(meals, period) {
  const mealList = meals
    .map((m) => m.date + " - " + m.food_name + ": " + m.calories + "cal, " + m.protein + "g protein, " + m.carbs + "g carbs, " + m.fat + "g fat")
    .join("\n");
  const totals = formatTotals(meals);
  const response = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 400,
    messages: [
      {
        role: "user",
        content: "My food diary for " + period + ":\n\n" + mealList + "\n\nTotals: " + totals.calories + " cal, " + totals.protein + "g protein, " + totals.carbs + "g carbs, " + totals.fat + "g fat, " + totals.fiber + "g fiber, " + totals.sugar + "g sugar\n\nGive me a brief friendly nutritional analysis. What am I doing well? One thing to improve. Under 100 words.",
      },
    ],
  });
  return response.content[0].text;
}

async function saveMeal(chatId, nutrition) {
  const { data, error } = await supabase
    .from("meals")
    .insert([{
      chat_id: chatId.toString(),
      food_name: nutrition.food_name,
      calories: nutrition.calories,
      protein: nutrition.protein,
      carbs: nutrition.carbs,
      fat: nutrition.fat,
      fiber: nutrition.fiber,
      sugar: nutrition.sugar,
      notes: nutrition.notes,
      logged_at: new Date().toISOString(),
      date: today(),
    }])
    .select()
    .single();
  if (error) throw new Error(error.message);
  return data;
}

async function getTodaysMeals(chatId) {
  const { data, error } = await supabase
    .from("meals").select("*")
    .eq("chat_id", chatId.toString())
    .eq("date", today())
    .order("logged_at", { ascending: true });
  if (error) throw new Error(error.message);
  return data;
}

async function getMealsForRange(chatId, startDate, endDate) {
  const { data, error } = await supabase
    .from("meals").select("*")
    .eq("chat_id", chatId.toString())
    .gte("date", startDate)
    .lte("date", endDate)
    .order("logged_at", { ascending: true });
  if (error) throw new Error(error.message);
  return data;
}

async function deleteLastMeal(chatId) {
  const { data } = await supabase
    .from("meals").select("id")
    .eq("chat_id", chatId.toString())
    .order("logged_at", { ascending: false })
    .limit(1).single();
  if (!data) return null;
  const { error } = await supabase.from("meals").delete().eq("id", data.id);
  if (error) throw new Error(error.message);
  return data.id;
}

function buildReplyMessage(nutrition) {
  if (nutrition.isBreakdown && nutrition.items && nutrition.items.length > 0) {
    const itemLines = nutrition.items
      .map((item) => "* *" + item.name + "* — " + item.calories + " kcal | P: " + item.protein + "g | C: " + item.carbs + "g | F: " + item.fat + "g")
      .join("\n");
    return (
      "Meal logged!\n\n" +
      "*Breakdown:*\n" + itemLines + "\n\n" +
      "*Total:*\n" +
      "Calories: " + nutrition.calories + " kcal\n" +
      "Protein: " + nutrition.protein + "g | Carbs: " + nutrition.carbs + "g | Fat: " + nutrition.fat + "g\n" +
      "Fiber: " + nutrition.fiber + "g | Sugar: " + nutrition.sugar + "g" +
      (nutrition.notes ? "\n\n_" + nutrition.notes + "_" : "") +
      "\n\nType /today to see your daily totals."
    );
  }
  return (
    "Meal logged!\n\n" +
    formatMeal(nutrition) +
    (nutrition.notes ? "\n\n_" + nutrition.notes + "_" : "") +
    "\n\nType /today to see your daily totals."
  );
}

// ============================================================================
// INTENT ROUTER
// ----------------------------------------------------------------------------
// Every plain text message is first classified into one intent, then routed
// to the matching handler below. Photos are always treated as food.
// ============================================================================

const INTENT_PROMPT = `You are an intent classifier for a personal health assistant.
Classify the user's message into EXACTLY ONE intent. Respond ONLY with raw JSON, no markdown: {"intent":"..."}

Intents:
- "log_food": describing food/drink they ate (e.g. "2 eggs and toast", "had a protein shake", "chicken bowl for lunch")
- "log_workout": describing exercise they did (e.g. "ran 3 miles", "chest day 45 min", "leg workout", "did yoga")
- "log_sleep": reporting how they slept (e.g. "slept 7 hours", "bed at 11 up at 6", "rough night, 5 hrs")
- "log_body": reporting body metrics (e.g. "weighed 180", "body fat 15%", "waist is 34")
- "set_goal": setting/changing a target (e.g. "set calorie goal to 2000", "I want 180g protein a day", "aim for 8 hours sleep")
- "question": asking about their OWN logged data (e.g. "how many calories today?", "how did I sleep this week?", "what did I eat yesterday?")
- "advice": asking for a recommendation or guidance (e.g. "what should I eat for dinner?", "am I on track?", "how can I hit my protein?")

Rules:
- Prefer the most specific match.
- If it mentions eating/drinking, it's log_food even if it also mentions other things.
- If truly ambiguous, use "log_food".`;

async function classifyIntent(text) {
  try {
    const response = await anthropic.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 50,
      messages: [{ role: "user", content: INTENT_PROMPT + '\n\nMessage: "' + text + '"' }],
    });
    const raw = extractJson(response.content[0].text);
    const valid = ["log_food", "log_workout", "log_sleep", "log_body", "set_goal", "question", "advice"];
    return valid.includes(raw.intent) ? raw.intent : "log_food";
  } catch (err) {
    console.error("Intent classify error:", err.message);
    return "log_food"; // safe fallback — preserves original behavior
  }
}

// ---- WORKOUT -------------------------------------------------------------
const WORKOUT_PROMPT = `Extract workout details from the message. Respond ONLY with raw JSON, no markdown:
{"workout_type":"short label e.g. Running, Chest Day, Yoga","duration":0,"calories_burned":0,"muscle_groups":"comma separated or empty","notes":""}
duration is whole minutes. calories_burned is a reasonable kcal estimate (integer). Estimate sensibly if not stated.`;

async function handleLogWorkout(chatId, text) {
  const response = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 300,
    messages: [{ role: "user", content: WORKOUT_PROMPT + '\n\nMessage: "' + text + '"' }],
  });
  const raw = extractJson(response.content[0].text);
  const workout = {
    chat_id: chatId.toString(),
    workout_type: raw.workout_type || "Workout",
    duration: Math.round(Number(raw.duration) || 0),
    calories_burned: Math.round(Number(raw.calories_burned) || 0),
    muscle_groups: raw.muscle_groups || "",
    notes: raw.notes || "",
    date: today(),
    logged_at: new Date().toISOString(),
  };
  const { error } = await supabase.from("workouts").insert([workout]);
  if (error) throw new Error(error.message);
  bot.sendMessage(chatId,
    "Workout logged! 💪\n\n*" + workout.workout_type + "*\n" +
    "Duration: " + workout.duration + " min\n" +
    "Calories burned: ~" + workout.calories_burned + " kcal" +
    (workout.muscle_groups ? "\nMuscles: " + workout.muscle_groups : "") +
    (workout.notes ? "\n\n_" + workout.notes + "_" : ""),
    { parse_mode: "Markdown" }
  );
}

// ---- SLEEP ---------------------------------------------------------------
const SLEEP_PROMPT = `Extract sleep details from the message. Respond ONLY with raw JSON, no markdown:
{"hours":0,"quality":null,"bedtime":"","notes":""}
hours may be a decimal (e.g. 7.5). quality is 1-10 only if the user implies it, else null. bedtime is free text or empty.`;

async function handleLogSleep(chatId, text) {
  const response = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 200,
    messages: [{ role: "user", content: SLEEP_PROMPT + '\n\nMessage: "' + text + '"' }],
  });
  const raw = extractJson(response.content[0].text);
  const sleep = {
    chat_id: chatId.toString(),
    hours: Number(raw.hours) || 0,
    quality: raw.quality == null ? null : Math.round(Number(raw.quality)),
    bedtime: raw.bedtime || "",
    notes: raw.notes || "",
    date: today(),
    logged_at: new Date().toISOString(),
  };
  const { error } = await supabase.from("sleep").insert([sleep]);
  if (error) throw new Error(error.message);
  bot.sendMessage(chatId,
    "Sleep logged! 😴\n\n" +
    "Hours: " + sleep.hours +
    (sleep.quality ? "\nQuality: " + sleep.quality + "/10" : "") +
    (sleep.bedtime ? "\nBedtime: " + sleep.bedtime : "") +
    (sleep.notes ? "\n\n_" + sleep.notes + "_" : ""),
    { parse_mode: "Markdown" }
  );
}

// ---- BODY METRICS --------------------------------------------------------
const BODY_PROMPT = `Extract body metrics from the message. Respond ONLY with raw JSON, no markdown:
{"weight":null,"body_fat":null,"measurements":{},"notes":""}
weight in lbs. body_fat as a percent number. measurements is an object of any others (e.g. {"waist":34}). Use null when not mentioned.`;

async function handleLogBody(chatId, text) {
  const response = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 250,
    messages: [{ role: "user", content: BODY_PROMPT + '\n\nMessage: "' + text + '"' }],
  });
  const raw = extractJson(response.content[0].text);
  const hasMeasurements = raw.measurements && Object.keys(raw.measurements).length > 0;
  const metric = {
    chat_id: chatId.toString(),
    weight: raw.weight == null ? null : Number(raw.weight),
    body_fat: raw.body_fat == null ? null : Number(raw.body_fat),
    measurements: hasMeasurements ? raw.measurements : null,
    notes: raw.notes || "",
    date: today(),
    logged_at: new Date().toISOString(),
  };
  const { error } = await supabase.from("body_metrics").insert([metric]);
  if (error) throw new Error(error.message);
  const parts = [];
  if (metric.weight != null) parts.push("Weight: " + metric.weight + " lbs");
  if (metric.body_fat != null) parts.push("Body fat: " + metric.body_fat + "%");
  if (hasMeasurements) {
    parts.push(Object.entries(raw.measurements).map(([k, v]) => k + ": " + v).join("\n"));
  }
  bot.sendMessage(chatId,
    "Body metrics logged! 📏\n\n" + (parts.join("\n") || "Saved.") +
    (metric.notes ? "\n\n_" + metric.notes + "_" : ""),
    { parse_mode: "Markdown" }
  );
}

// ---- GOALS ---------------------------------------------------------------
const GOAL_PROMPT = `Extract the health targets the user wants to set. Respond ONLY with raw JSON, no markdown. Use null for anything not mentioned:
{"calorie_target":null,"protein_target":null,"carbs_target":null,"fat_target":null,"fiber_target":null,"sleep_target":null,"water_target":null}
calorie_target in kcal, macro targets in grams, sleep_target in hours, water_target in oz. All numbers.`;

async function getGoals(chatId) {
  const { data } = await supabase
    .from("goals").select("*")
    .eq("chat_id", chatId.toString())
    .order("updated_at", { ascending: false })
    .limit(1).single();
  return data || null;
}

async function handleSetGoal(chatId, text) {
  const response = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 200,
    messages: [{ role: "user", content: GOAL_PROMPT + '\n\nMessage: "' + text + '"' }],
  });
  const raw = extractJson(response.content[0].text);
  const fields = ["calorie_target", "protein_target", "carbs_target", "fat_target", "fiber_target", "sleep_target", "water_target"];

  const existing = await getGoals(chatId);
  const merged = { chat_id: chatId.toString(), updated_at: new Date().toISOString() };
  fields.forEach((f) => {
    if (raw[f] != null) merged[f] = Number(raw[f]);
    else if (existing && existing[f] != null) merged[f] = existing[f];
  });

  if (existing) {
    const { error } = await supabase.from("goals").update(merged).eq("id", existing.id);
    if (error) throw new Error(error.message);
  } else {
    const { error } = await supabase.from("goals").insert([merged]);
    if (error) throw new Error(error.message);
  }

  const labels = {
    calorie_target: "Calories", protein_target: "Protein (g)", carbs_target: "Carbs (g)",
    fat_target: "Fat (g)", fiber_target: "Fiber (g)", sleep_target: "Sleep (hrs)", water_target: "Water (oz)",
  };
  const lines = fields.filter((f) => merged[f] != null).map((f) => labels[f] + ": " + merged[f]);
  bot.sendMessage(chatId,
    "Goals updated! 🎯\n\n" + (lines.join("\n") || "No targets recognized — try \"set my calorie goal to 2000\"."),
    { parse_mode: "Markdown" }
  );
}

// ---- CONTEXT for questions / advice -------------------------------------
// Pulls a compact snapshot of the user's recent data to ground Claude's reply.
async function getRecentContext(chatId) {
  const id = chatId.toString();
  const end = today();
  const startD = new Date();
  startD.setDate(startD.getDate() - 6);
  const start = startD.toISOString().split("T")[0];

  const [mealsRes, workoutsRes, sleepRes, bodyRes] = await Promise.all([
    supabase.from("meals").select("*").eq("chat_id", id).gte("date", start).lte("date", end).order("logged_at", { ascending: true }),
    supabase.from("workouts").select("*").eq("chat_id", id).gte("date", start).lte("date", end).order("logged_at", { ascending: true }),
    supabase.from("sleep").select("*").eq("chat_id", id).gte("date", start).lte("date", end).order("logged_at", { ascending: true }),
    supabase.from("body_metrics").select("*").eq("chat_id", id).order("logged_at", { ascending: false }).limit(3),
  ]);
  const goals = await getGoals(chatId);

  const meals = mealsRes.data || [];
  const todaysMeals = meals.filter((m) => m.date === end);
  const todayTotals = formatTotals(todaysMeals);

  let ctx = "Today is " + end + " (Eastern).\n\n";

  ctx += "TODAY'S MEALS:\n" + (todaysMeals.length
    ? todaysMeals.map((m) => "- " + m.food_name + ": " + m.calories + " kcal, " + m.protein + "g protein").join("\n") +
      "\nToday totals: " + todayTotals.calories + " kcal, " + todayTotals.protein + "g protein, " + todayTotals.carbs + "g carbs, " + todayTotals.fat + "g fat"
    : "none logged today") + "\n\n";

  ctx += "LAST 7 DAYS MEALS (count " + meals.length + "):\n" + (meals.length
    ? Object.entries(meals.reduce((a, m) => { a[m.date] = (a[m.date] || 0) + (m.calories || 0); return a; }, {}))
        .map(([d, c]) => "- " + d + ": " + c + " kcal").join("\n")
    : "none") + "\n\n";

  ctx += "WORKOUTS (7d):\n" + ((workoutsRes.data || []).length
    ? workoutsRes.data.map((w) => "- " + w.date + ": " + w.workout_type + ", " + w.duration + " min, ~" + w.calories_burned + " kcal").join("\n")
    : "none logged") + "\n\n";

  ctx += "SLEEP (7d):\n" + ((sleepRes.data || []).length
    ? sleepRes.data.map((s) => "- " + s.date + ": " + s.hours + " hrs" + (s.quality ? " (quality " + s.quality + "/10)" : "")).join("\n")
    : "none logged") + "\n\n";

  ctx += "RECENT BODY METRICS:\n" + ((bodyRes.data || []).length
    ? bodyRes.data.map((b) => "- " + b.date + ": " + (b.weight != null ? b.weight + " lbs" : "") + (b.body_fat != null ? " " + b.body_fat + "% bf" : "")).join("\n")
    : "none logged") + "\n\n";

  ctx += "GOALS:\n" + (goals
    ? ["calorie_target", "protein_target", "carbs_target", "fat_target", "fiber_target", "sleep_target", "water_target"]
        .filter((f) => goals[f] != null).map((f) => "- " + f + ": " + goals[f]).join("\n") || "none set"
    : "none set");

  return ctx;
}

async function handleQuestion(chatId, text) {
  const context = await getRecentContext(chatId);
  const response = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 400,
    messages: [{
      role: "user",
      content: "You are the user's friendly personal health assistant. Use ONLY the data below to answer their question accurately. If the data doesn't cover it, say so plainly. Keep it conversational and under 120 words. Don't invent numbers.\n\n=== USER DATA ===\n" +
        context + "\n=== END DATA ===\n\nQuestion: " + text,
    }],
  });
  bot.sendMessage(chatId, response.content[0].text);
}

async function handleAdvice(chatId, text) {
  const context = await getRecentContext(chatId);
  const response = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 450,
    messages: [{
      role: "user",
      content: "You are the user's supportive personal health coach. Give specific, encouraging, practical advice grounded in their actual data below. Reference their goals where relevant. Be realistic and kind — never extreme. Keep it under 140 words.\n\n=== USER DATA ===\n" +
        context + "\n=== END DATA ===\n\nRequest: " + text,
    }],
  });
  bot.sendMessage(chatId, response.content[0].text);
}

async function handleLogFood(chatId, text) {
  const nutrition = await analyzeFoodText(text);
  await saveMeal(chatId, nutrition);
  bot.sendMessage(chatId, buildReplyMessage(nutrition), { parse_mode: "Markdown" });
}

// ============================================================================
// COMMANDS
// ============================================================================

bot.onText(/\/start/, (msg) => {
  bot.sendMessage(msg.chat.id,
    "Welcome to your *Health Assistant*!\n\nJust talk to me naturally:\n\n📸 Send a *photo* of food to log it\n🍳 *\"2 eggs and toast\"* — log food\n💪 *\"ran 3 miles\"* — log a workout\n😴 *\"slept 7 hours\"* — log sleep\n📏 *\"weighed 180\"* — log body metrics\n🎯 *\"set my calorie goal to 2000\"* — set a target\n❓ *\"how many calories today?\"* — ask about your data\n💡 *\"what should I eat for dinner?\"* — get advice\n\n*Commands:*\n/today /week /undo /help",
    { parse_mode: "Markdown" }
  );
});

bot.onText(/\/help/, (msg) => {
  bot.sendMessage(msg.chat.id,
    "*I understand natural language.* Try:\n\n🍳 \"chicken bowl for lunch\" (food)\n💪 \"chest day 45 min\" (workout)\n😴 \"bed at 11, up at 6\" (sleep)\n📏 \"body fat 15%\" (metrics)\n🎯 \"aim for 180g protein\" (goal)\n❓ \"how did I sleep this week?\" (question)\n💡 \"am I on track today?\" (advice)\n\n*Commands:*\n/today - today's meals and totals\n/week - week breakdown + AI analysis\n/undo - remove last logged meal\n/help - this message",
    { parse_mode: "Markdown" }
  );
});

bot.onText(/\/today/, async (msg) => {
  const chatId = msg.chat.id;
  bot.sendChatAction(chatId, "typing");
  try {
    const meals = await getTodaysMeals(chatId);
    if (meals.length === 0) {
      return bot.sendMessage(chatId, "No meals logged today yet! Send a photo or describe what you ate.");
    }
    const totals = formatTotals(meals);
    const mealLines = meals.map((m, i) => (i + 1) + ". " + m.food_name + " - " + m.calories + " kcal").join("\n");
    bot.sendMessage(chatId,
      "*Today's Food Log*\n\n" + mealLines + "\n\n*Daily Totals:*\nCalories: " + totals.calories + " kcal\nProtein: " + totals.protein + "g | Carbs: " + totals.carbs + "g | Fat: " + totals.fat + "g\nFiber: " + totals.fiber + "g | Sugar: " + totals.sugar + "g",
      { parse_mode: "Markdown" }
    );
  } catch (err) {
    console.error(err);
    bot.sendMessage(chatId, "Error fetching today's meals. Try again!");
  }
});

bot.onText(/\/week/, async (msg) => {
  const chatId = msg.chat.id;
  bot.sendChatAction(chatId, "typing");
  try {
    const endDate = today();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 6);
    const startDateStr = startDate.toISOString().split("T")[0];
    const meals = await getMealsForRange(chatId, startDateStr, endDate);
    if (meals.length === 0) {
      return bot.sendMessage(chatId, "No meals logged this week yet!");
    }
    const totals = formatTotals(meals);
    const avgCalories = Math.round(totals.calories / 7);
    const summary = await getAISummary(meals, "this week");
    bot.sendMessage(chatId,
      "*This Week's Summary* (" + startDateStr + " to " + endDate + ")\n\n*Totals:*\nCalories: " + totals.calories + " kcal (avg " + avgCalories + "/day)\nProtein: " + totals.protein + "g | Carbs: " + totals.carbs + "g | Fat: " + totals.fat + "g\nFiber: " + totals.fiber + "g | Sugar: " + totals.sugar + "g\nMeals logged: " + meals.length + "\n\n*AI Analysis:*\n" + summary,
      { parse_mode: "Markdown" }
    );
  } catch (err) {
    console.error(err);
    bot.sendMessage(chatId, "Error fetching weekly data. Try again!");
  }
});

bot.onText(/\/undo/, async (msg) => {
  const chatId = msg.chat.id;
  try {
    const deleted = await deleteLastMeal(chatId);
    if (!deleted) return bot.sendMessage(chatId, "No meals to undo!");
    bot.sendMessage(chatId, "Last meal removed!");
  } catch (err) {
    console.error(err);
    bot.sendMessage(chatId, "Error removing last meal. Try again!");
  }
});

// Photos are always food.
bot.on("photo", async (msg) => {
  const chatId = msg.chat.id;
  bot.sendChatAction(chatId, "typing");
  bot.sendMessage(chatId, "Got your photo! Analyzing the food...");
  try {
    const photo = msg.photo[msg.photo.length - 1];
    const file = await bot.getFile(photo.file_id);
    const imageUrl = "https://api.telegram.org/file/bot" + process.env.TELEGRAM_BOT_TOKEN + "/" + file.file_path;
    const nutrition = await analyzeFoodImage(imageUrl);
    await saveMeal(chatId, nutrition);
    bot.sendMessage(chatId, buildReplyMessage(nutrition), { parse_mode: "Markdown" });
  } catch (err) {
    console.error("Photo error:", err.message);
    bot.sendMessage(chatId, "Couldn't analyze that photo. Try a clearer shot or describe the meal in text!");
  }
});

// All plain text messages go through the intent router.
bot.on("message", async (msg) => {
  if (msg.text && msg.text.startsWith("/")) return;
  if (msg.photo) return;
  if (!msg.text) return;
  const chatId = msg.chat.id;
  bot.sendChatAction(chatId, "typing");

  try {
    const intent = await classifyIntent(msg.text);
    switch (intent) {
      case "log_workout": return await handleLogWorkout(chatId, msg.text);
      case "log_sleep":   return await handleLogSleep(chatId, msg.text);
      case "log_body":    return await handleLogBody(chatId, msg.text);
      case "set_goal":    return await handleSetGoal(chatId, msg.text);
      case "question":    return await handleQuestion(chatId, msg.text);
      case "advice":      return await handleAdvice(chatId, msg.text);
      case "log_food":
      default:            return await handleLogFood(chatId, msg.text);
    }
  } catch (err) {
    console.error("Router error:", err.message);
    bot.sendMessage(chatId, "Hmm, I couldn't process that. Try rephrasing, or be more specific!");
  }
});

http.createServer((req, res) => res.end("OK")).listen(process.env.PORT || 3000);
// Self-ping to prevent Render free tier spin-down
const RENDER_URL = process.env.RENDER_URL;
if (RENDER_URL) {
  setInterval(() => {
    http.get(RENDER_URL, (res) => {
      console.log("Self-ping:", res.statusCode);
    }).on("error", (err) => {
      console.log("Ping error:", err.message);
    });
  }, 14 * 60 * 1000); // every 14 minutes
}
console.log("Health Assistant Bot is running with Claude AI...");
