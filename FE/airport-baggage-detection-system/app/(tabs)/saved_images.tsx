import { useAuth } from "@/context/AuthenticationContext";
import { Ionicons } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";
import React, { useEffect, useState } from "react";
import {
  FlatList,
  Image,
  RefreshControl,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

const API_URL = "http://192.168.1.134:5000/api";

interface MyImage {
  _id: string;
  filename: string;
  createdAt: string;
}
export default function MyImagesScreen() {
  const { getUserImages, user } = useAuth();
  const navigation = useNavigation();
  const [images, setImages] = useState<MyImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadImages = async () => {
    try {
      setLoading(true);
      const userImages = await getUserImages();
      setImages(userImages);
    } catch (error) {
      console.error("Failed to load images", error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadImages();
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    loadImages();
  };

  const renderImageItem = ({ item }: { item: MyImage }) => (
    <View style={styles.imageCard}>
      <Image
        source={{ uri: `${API_URL}/images/${item.filename}` }}
        style={styles.image}
        resizeMode="cover"
      />
      <View style={styles.imageInfo}>
        <Text style={styles.imageDate}>
          {new Date(item.createdAt).toLocaleDateString()}
        </Text>
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>My Stickers</Text>
      </View>

      {loading && images.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>Loading your stickers...</Text>
        </View>
      ) : images.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Ionicons name="images-outline" size={80} color="#666" />
          <Text style={styles.emptyText}>
            You haven't created any stickers yet
          </Text>
          <TouchableOpacity
            style={styles.createButton}
            onPress={() => navigation.navigate("(tabs)" as never)}
          >
            <Text style={styles.createButtonText}>Create a Sticker</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={images}
          renderItem={renderImageItem}
          keyExtractor={(item) => item._id}
          contentContainerStyle={styles.imageList}
          numColumns={2}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              colors={["#ffd33d"]}
              tintColor="#ffd33d"
            />
          }
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#25292e",
  },
  header: {
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: "#444",
  },
  title: {
    fontSize: 20,
    fontWeight: "bold",
    color: "white",
  },
  imageList: {
    padding: 10,
  },
  imageCard: {
    flex: 1,
    margin: 8,
    borderRadius: 10,
    overflow: "hidden",
    backgroundColor: "#333",
  },
  image: {
    width: "100%",
    height: 150,
  },
  imageInfo: {
    padding: 8,
  },
  imageDate: {
    color: "#ccc",
    fontSize: 12,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  emptyText: {
    color: "#999",
    fontSize: 16,
    textAlign: "center",
    marginTop: 16,
  },
  createButton: {
    marginTop: 20,
    backgroundColor: "#ffd33d",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  createButtonText: {
    color: "#25292e",
    fontWeight: "bold",
  },
});
