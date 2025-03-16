import Button from "@/components/Buttons";
import ImageViewer from "@/components/ImageViewer";
import * as ImagePicker from "expo-image-picker";
import { useState } from "react";
import { StyleSheet, View } from "react-native";

const PlaceholderImage = require("../../assets/images/image.png");

export default function Index() {
  const [selectedImage, setSelectedImage] = useState<string | undefined>(
    undefined
  );
  const pcikImageAsync = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      console.log(result);
    } else {
      alert("Please select an image");
    }
  };
  return (
    <View style={styles.container}>
      <View style={styles.imageContainer}>
        {/* hardcoded <Image source={PlaceholderImage} style={styles.image} /> */}
        <ImageViewer imgSource={selectedImage || PlaceholderImage} />
      </View>
      <View style={styles.footerContainer}>
        <Button onPress={pcikImageAsync} label="Choose photo" theme="primary" />
        <Button label="Use this photo" />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#25292e",
  },
  text: {
    color: "white",
  },
  button: {
    fontSize: 20,
    textDecorationLine: "underline",
    color: "#fff",
  },
  image: {
    width: 320,
    height: 440,
    borderRadius: 18,
  },
  imageContainer: {
    flex: 1,
  },
  footerContainer: {
    flex: 1 / 3,
    alignItems: "center",
  },
});
