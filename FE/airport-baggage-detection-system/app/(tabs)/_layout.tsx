import { Ionicons } from "@expo/vector-icons";
import { Tabs } from "expo-router";
import { LogBox } from "react-native";

LogBox.ignoreAllLogs(true);
export default function RootLayout() {
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: "ffd33d",
        headerStyle: { backgroundColor: "#2529" },
        headerShadowVisible: false,
        headerTintColor: "#fff",
        tabBarStyle: {
          backgroundColor: "#25292e",
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          headerTitle: "Home",
          tabBarIcon: ({ focused, color }) => (
            <Ionicons
              name={focused ? "home-sharp" : "home-outline"}
              color={color}
              size={30}
            />
          ),
        }}
      />
      <Tabs.Screen
        name="+not-found"
        options={{
          headerTitle: "+not-found",
        }}
      />
    </Tabs>
  );
}
