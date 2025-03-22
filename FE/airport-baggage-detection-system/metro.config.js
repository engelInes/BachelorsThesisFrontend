const { getDefaultConfig } = require("@react-native/metro-config");
const path = require("path");

/** @type {import('@react-native/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// If you need support for .cjs files, keep this line
config.resolver.sourceExts.push("cjs");

config.transformer = {
  ...config.transformer,
  unstable_allowRequireContext: true,
};

process.env.EXPO_ROUTER_APP_ROOT = "./app";

module.exports = config;
