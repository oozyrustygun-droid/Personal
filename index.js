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
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("No JSON found in response");
  const raw = JSON.parse(match[0]);
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
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("No JSON found");
  const raw = JSON.parse(match[0]);

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
      ...parseNutrition(JSON.stringify(raw.total)),
    };
  }

  return { isBreakdown: false, ...parseNutrition(text) };
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

bot.onText(/\/start/, (msg) => {
  bot.sendMessage(msg.chat.id,
    "Welcome to your *Calorie Tracker*!\n\nHow to use me:\n\nSend a *photo* of your food and I will analyze and log it\nOr *describe your food* in text and I will estimate and log it\n\n*Commands:*\n/today - see today's meals and totals\n/week - see this week's summary\n/undo - delete your last logged meal\n/help - show this message",
    { parse_mode: "Markdown" }
  );
});

bot.onText(/\/help/, (msg) => {
  bot.sendMessage(msg.chat.id,
    "*Commands:*\n/today - today's meals and calorie totals\n/week - this week's breakdown plus AI analysis\n/undo - remove last logged meal\n/help - show this message\n\n*To log food:*\nSend a photo of your meal\nOr type what you ate e.g. 2 scrambled eggs and toast",
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

bot.on("message", async (msg) => {
  if (msg.text && msg.text.startsWith("/")) return;
  if (msg.photo) return;
  if (!msg.text) return;
  const chatId = msg.chat.id;
  bot.sendChatAction(chatId, "typing");
  try {
    const nutrition = await analyzeFoodText(msg.text);
    await saveMeal(chatId, nutrition);
    bot.sendMessage(chatId, buildReplyMessage(nutrition), { parse_mode: "Markdown" });
  } catch (err) {
    console.error("Text error:", err.message);
    bot.sendMessage(chatId, "Couldn't log that meal. Try being more specific, like: 2 eggs, toast, and orange juice");
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
console.log("Calorie Tracker Bot is running with Claude AI...");
