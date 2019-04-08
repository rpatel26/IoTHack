//
//  ViewController.swift
//  IotHack
//
//  Created by Ravi Patel on 3/24/19.
//  Copyright © 2019 IoTHack. All rights reserved.
//

////////////////////////////////////////////////////
// API KEY: AIzaSyAXu0OYZD9_R0nyfVLCRSmGbFKcd6z9H_U
////////////////////////////////////////////////////

import UIKit
import Darwin
import FirebaseDatabase
import GoogleMaps

class ViewController: UIViewController, CLLocationManagerDelegate {

    var ref: DatabaseReference!
    var myMap: GMSMapView!
    var markers = [GMSMarker]()
    var currentLatitude: Double!
    var currentLongitude: Double!
    var currentLocation: CLLocationCoordinate2D!
    var currentMarker: GMSMarker!
    
    override func viewDidLoad() {
        super.viewDidLoad()

        ref = Database.database().reference()
        
        currentLatitude = 32.869131
        currentLongitude = -117.217818
        
        GMSServices.provideAPIKey("AIzaSyAXu0OYZD9_R0nyfVLCRSmGbFKcd6z9H_U")
        let camera = GMSCameraPosition.camera(withLatitude: currentLatitude, longitude:  currentLongitude, zoom: 1)
//        let mapView = GMSMapView.map(withFrame: CGRect.zero, camera: camera)
        myMap = GMSMapView.map(withFrame: CGRect.zero, camera: camera)
        self.view = myMap
        myMap.settings.compassButton = true
        myMap.settings.myLocationButton = true
        myMap.isMyLocationEnabled = true
        
        myMap.addObserver(self, forKeyPath: "myLocation", options: .new, context: nil)
        
        currentLocation = CLLocationCoordinate2DMake(currentLatitude, currentLongitude)
        currentMarker = GMSMarker(position: currentLocation)
        currentMarker.title = "Costa Verde"
        currentMarker.map = myMap
        currentMarker.icon = GMSMarker.markerImage(with: UIColor.darkGray)
        
        

        // Read from database in realtime
        ref.child("bots").observe(.value, with: { (snapshot) in
            let value = snapshot.value as? NSDictionary
            
            let keys = value?.allKeys ?? []
            let values = value?.allValues ?? []

            for i in 0..<self.markers.count{
                self.markers[i].map = nil
            }

            self.markers.removeAll()

//            print("Realtime database updated")
            for i in 0..<keys.count{
//                print("keys = ", keys[i])
//                print("value = ", value![keys[i]] ?? ".............")
                let location = values[i] as! NSDictionary
                let latitude = location.value(forKey: "latitude") ?? 0.0
                let longitude = location.value(forKey: "longitude") ?? 0.0
                let markerLocation = CLLocationCoordinate2DMake(latitude as! CLLocationDegrees, longitude as! CLLocationDegrees)
                let marker = GMSMarker(position: markerLocation)
                marker.title = keys[i] as? String
                marker.map = self.myMap
                self.markers.append(marker)

            }
        }) { (err) in
            print("error reading from database.......error = ", err)
        }
        
        navigationItem.rightBarButtonItem = UIBarButtonItem(title: "Request", style: .done, target: self, action: #selector(request))
    }
    
    @objc func request(){
        print("Request...")
        
        var closest_index = -1
        var closest_dist = Double.greatestFiniteMagnitude
        
        for i in 0..<self.markers.count{
            let dist = getDistance(latitude: self.markers[i].position.latitude, longitude: self.markers[i].position.longitude)
            
            if dist < closest_dist {
                closest_index = i
                closest_dist = dist
            }
        }
        
        if closest_index != -1{
            print("Closest bot = ", self.markers[closest_index].title!)
        }
        else{
            print("No closest bot detected....")
        }
        
    }

    func getDistance(latitude: Double, longitude: Double) -> Double{
        let dist1 = self.currentLatitude - latitude
        let dist2 = self.currentLongitude - longitude
        return Double(exactly: sqrt(dist1 * dist1 + dist2 * dist2))!
    }
}
