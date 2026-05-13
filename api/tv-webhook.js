const { Redis } = require("@upstash/redis");

function getRedis() {
  const url =
    process.env.UPSTASH_REDIS_REST_URL ||
    process.env.KV_REST_API_URL;

  const token =
    process.env.UPSTASH_REDIS_REST_TOKEN ||
    process.env.KV_REST_API_TOKEN;

  if (!url || !token) {
    throw new Error("Redis is not connected. Add Upstash Redis / Vercel Redis to this Vercel project.");
  }

  return new Redis({ url, token });
}

function toNumber(value) {
  if (value === null || value === undefined || value === "" || value === "na") {
    return null;
  }

  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function cleanString(value) {
  return String(value ?? "").trim();
}

function tradingViewUrl(tvSymbol) {
  return "https://www.tradingview.com/chart/?symbol=" + encodeURIComponent(tvSymbol);
}

function sideForDecision(decision) {
  if ([
    "BUY",
    "BUY CONFIRMED",
    "BUY / EXIT SHORT",
    "GET READY TO BUY",
    "LEAN BUY",
    "HOLD LONG",
    "TRAIL LONG"
  ].includes(decision)) {
    return "BUY";
  }

  if ([
    "SELL",
    "SELL CONFIRMED",
    "SELL / EXIT LONG",
    "GET READY TO SELL",
    "LEAN SELL",
    "HOLD SHORT",
    "TRAIL SHORT"
  ].includes(decision)) {
    return "SELL";
  }

  if ([
    "SELL / EXIT LONG",
    "BUY / EXIT SHORT"
  ].includes(decision)) {
    return "EXIT";
  }

  if (decision === "ERROR") {
    return "ERROR";
  }

  return "WAIT";
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      ok: false,
      error: "Use POST only."
    });
  }

  try {
    let data = req.body;

    if (typeof data === "string") {
      data = JSON.parse(data);
    }

    if (!data || typeof data !== "object") {
      throw new Error("Invalid JSON body from TradingView.");
    }

    const expectedSecret = process.env.TV_WEBHOOK_SECRET || "";

    if (expectedSecret && data.secret !== expectedSecret) {
      return res.status(401).json({
        ok: false,
        error: "Bad webhook secret."
      });
    }

    const exchange = cleanString(data.exchange || data.prefix || "UNKNOWN").toUpperCase();
    const symbol = cleanString(data.symbol || data.ticker || "UNKNOWN").toUpperCase();
    const timeframe = cleanString(data.timeframe || data.interval || "UNKNOWN");

    const tvSymbol =
      cleanString(data.tv_symbol || data.tradingview_symbol) ||
      `${exchange}:${symbol}`;

    const decision = cleanString(data.decision || data.action || "WAIT");
    const position = cleanString(data.position || "FLAT");

    const row = {
      id: `${tvSymbol}:${timeframe}`,
      symbol,
      exchange,
      tv_symbol: tvSymbol,
      tradingview_symbol: tvSymbol,
      tradingview_url: tradingViewUrl(tvSymbol),
      timeframe,

      decision,
      side: sideForDecision(decision),
      position,

      buy_percent: toNumber(data.buy_percent),
      sell_percent: toNumber(data.sell_percent),
      edge_percent: toNumber(data.edge_percent),
      confidence: toNumber(data.confidence),

      last_price: toNumber(data.last_price || data.close),
      entry: toNumber(data.entry),
      stop: toNumber(data.stop),
      tp1: toNumber(data.tp1),
      rsi: toNumber(data.rsi),

      market: cleanString(data.market || ""),
      why: cleanString(data.why || data.reason || ""),

      bar_time: toNumber(data.bar_time),
      bar_time_iso: data.bar_time ? new Date(Number(data.bar_time)).toISOString() : null,

      sent_at: toNumber(data.sent_at),
      received_at: new Date().toISOString()
    };

    const redis = getRedis();

    await redis.hset("tv:latest-signals", {
      [row.id]: JSON.stringify(row)
    });

    await redis.set("tv:last-updated", row.received_at);

    return res.status(200).json({
      ok: true,
      saved: row.id,
      decision: row.decision
    });

  } catch (error) {
    return res.status(400).json({
      ok: false,
      error: error.message
    });
  }
};
