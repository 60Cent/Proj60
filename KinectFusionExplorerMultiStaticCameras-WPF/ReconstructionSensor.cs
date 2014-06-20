// -----------------------------------------------------------------------
// <copyright file="ReconstructionSensor.cs" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// -----------------------------------------------------------------------

namespace Microsoft.Samples.Kinect.KinectFusionExplorer
{
    using System;
    using System.IO;
    using System.Threading.Tasks;
    using System.Windows;
    using System.Windows.Media.Media3D;
    using System.Xml.Serialization;
    using Microsoft.Kinect;
    using Microsoft.Kinect.Toolkit.Fusion;

    /// <summary>
    /// This basic wrapper class encapsulates a Kinect Sensor and processing resources required for creating a multi-camera Reconstruction
    /// </summary>
    public class ReconstructionSensor : IDisposable
    {
        #region Fields

        /// <summary>
        /// started is set true when the camera is running
        /// </summary>
        private bool started = false;

        /// <summary>
        /// Track whether Dispose has been called
        /// </summary>
        private bool disposed = false;

        /// <summary>
        /// The sensor instance
        /// </summary>
        private KinectSensor sensor;

        /// <summary>
        /// Format of depth image to use DepthImageFormat. Undefined if not used
        /// </summary>
        private DepthImageFormat depthFormat;

        /// <summary>
        /// Format of depth image to use ColorImageFormat. Undefined if not used
        /// </summary>
        private ColorImageFormat colorFormat;

        /// <summary>
        /// Image DepthWidth of depth frame
        /// </summary>
        private int depthWidth = 0;

        /// <summary>
        /// Image DepthHeight of depth frame
        /// </summary>
        private int depthHeight = 0;

        /// <summary>
        /// Image DepthWidth of color frame
        /// </summary>
        private int colorWidth = 0;

        /// <summary>
        /// Image DepthHeight of color frame
        /// </summary>
        private int colorHeight = 0;

        /// <summary>
        /// Intermediate storage for the extended depth data received from the camera in the current frame
        /// </summary>
        private DepthImagePixel[] depthImagePixels;

        /// <summary>
        /// Intermediate storage for the color data received from the camera in 32bit color
        /// </summary>
        private byte[] colorImagePixels;

        /// <summary>
        /// Mapping of depth pixels into color image
        /// </summary>
        private ColorImagePoint[] colorCoordinates;

        /// <summary>
        /// Mapped color pixels in depth frame of reference
        /// </summary>
        private int[] mappedColorPixels;

        /// <summary>
        /// The coordinate mapper to convert between depth and color frames of reference
        /// </summary>
        private CoordinateMapper mapper;

        /// <summary>
        /// Kinect Reconstruction Sensor Control
        /// </summary>
        private ReconstructionSensorControl reconstructionSensorControl;

        /// <summary>
        /// Kinect Reconstruction Graphics Camera
        /// </summary>
        private GraphicsCamera reconstructionSensorCamera;

        /// <summary>
        /// Synchronization object for color access
        /// </summary>
        private object colorLock = new object();

        #endregion

        /// <summary>
        /// Initializes a new instance of the <see cref="ReconstructionSensor"/> class.
        /// </summary>
        /// <param name="sensorInstance">The Kinect sensor to encapsulate.</param>
        public ReconstructionSensor(KinectSensor sensorInstance)
        {
            this.sensor = sensorInstance;
            this.reconstructionSensorControl = new ReconstructionSensorControl();
            this.reconstructionSensorControl.RequireResetReconstructionEvent += this.OnResetReconstruction;
            this.reconstructionSensorControl.RequireRenderReconstructionEvent += this.OnRenderReconstruction;
            this.reconstructionSensorControl.SetTransformationEvent += this.OnSetCameraTransformation;
            this.reconstructionSensorControl.SetCaptureColorEvent += this.OnSetCaptureColor;
        }

        /// <summary>
        /// Finalizes an instance of the ReconstructionSensor class.
        /// This destructor will run only if the Dispose method does not get called.
        /// </summary>
        ~ReconstructionSensor()
        {
            this.Dispose(false);
        }

        #region Properties

        /// <summary>
        /// Delegate for Event for depth ready and resources initialized if necessary
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        internal delegate void DepthFrameEventHandler(object sender, EventArgs e);

        /// <summary>
        /// Delegate for Event for color ready and resources initialized if necessary
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        internal delegate void ColorFrameEventHandler(object sender, EventArgs e);

        /// <summary>
        /// Delegate for Event for reset required on parameter change
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        internal delegate void RequireResetEventHandler(object sender, EventArgs e);

        /// <summary>
        /// Delegate for Event for render required on parameter change
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        internal delegate void RequireRenderEventHandler(object sender, EventArgs e);

        /// <summary>
        /// Delegate for Event for camera transform changed from UI
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        internal delegate void SensorTransformChangedEventHandler(object sender, EventArgs e);

        /// <summary>
        /// Delegate for Event for setting capture color for all sensors
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        internal delegate void AllSetCaptureColorEventHandler(object sender, EventArgs e);

        /// <summary>
        /// Event for depth ready and resources initialized if necessary
        /// </summary>
        internal event DepthFrameEventHandler DepthFrameReady;

        /// <summary>
        /// Event for color ready and resources initialized if necessary
        /// </summary>
        internal event ColorFrameEventHandler ColorFrameReady;

        /// <summary>
        /// Event for reconstruction reset required on parameter change
        /// </summary>
        internal event RequireResetEventHandler RequireResetEvent;

        /// <summary>
        /// Event for reconstruction reset required on parameter change
        /// </summary>
        internal event RequireRenderEventHandler RequireRenderEvent;

        /// <summary>
        /// Event for camera transform changed from UI
        /// </summary>
        internal event SensorTransformChangedEventHandler SensorTransformationChanged;

        /// <summary>
        /// Event for setting capture color for all sensors
        /// </summary>
        internal event AllSetCaptureColorEventHandler AllSetCaptureColorEvent;

        /// <summary>
        /// Gets or sets the timestamp of the current frame
        /// </summary>
        public long FrameTimestamp { get; set; }

        /// <summary>
        /// Gets or sets the depth in meters in float format converted from DepthImagePixels
        /// </summary>
        public FusionFloatImageFrame DepthFloatFrame { get; set; }

        /// <summary>
        /// Gets or sets the point cloud frame calculated either from depth or from the Reconstruction
        /// </summary>
        public FusionPointCloudImageFrame PointCloudFrame { get; set; }

        /// <summary>
        /// Gets or sets the shaded surface frame from shading point cloud frame
        /// </summary>
        public FusionColorImageFrame ShadedSurfaceFrame { get; set; }

        /// <summary>
        /// Gets or sets the shaded surface normals frame from shading point cloud frame
        /// </summary>
        public FusionColorImageFrame ShadedSurfaceNormalsFrame { get; set; }

        /// <summary>
        /// Gets or sets the per-pixel alignment values
        /// </summary>
        public FusionFloatImageFrame DeltaFromReferenceFrame { get; set; }

        /// <summary>
        /// Kinect color mapped into depth frame
        /// </summary>
        public FusionColorImageFrame MappedColorFrame { get; set; }

        /// <summary>
        /// Gets or sets the alignment energy from for current frame 
        /// </summary>
        public float AlignmentEnergy { get; set; }

        /// <summary>
        /// Gets or sets the status message
        /// </summary>
        public string StatusMessage { get; set; }

        /// <summary>
        /// Gets the Kinect Reconstruction Sensor Control
        /// </summary>
        public ReconstructionSensorControl ReconSensorControl
        {
            get
            {
                return this.reconstructionSensorControl;
            }
        }

        /// <summary>
        /// Gets the Kinect Reconstruction Sensor Camera
        /// </summary>
        public GraphicsCamera ReconCamera
        {
            get
            {
                return this.reconstructionSensorCamera;
            }
        }

        /// <summary>
        /// Gets the active Kinect Sensor
        /// </summary>
        public KinectSensor Sensor
        {
            get
            {
                return this.sensor;
            }
        }

        /// <summary>
        /// Gets the Kinect Id
        /// </summary>
        public string DeviceConnectionId
        {
            get
            {
                if (this.sensor != null && this.sensor.DeviceConnectionId != null)
                {
                    return this.sensor.DeviceConnectionId;
                }
                else
                {
                    return string.Empty;
                }
            }
        }

        /// <summary>
        /// Gets the format of depth image to use DepthImageFormat. Undefined if not used
        /// </summary>
        public DepthImageFormat DepthFormat
        {
            get
            {
                return this.depthFormat;
            }
        }

        /// <summary>
        /// Gets the format of depth image to use ColorImageFormat. Undefined if not used
        /// </summary>
        public ColorImageFormat ColorFormat
        {
            get
            {
                return this.colorFormat;
            }
        }

        /// <summary>
        /// Gets the image Width of depth frame
        /// </summary>
        public int DepthWidth
        {
            get
            {
                return this.depthWidth;
            }
        }

        /// <summary>
        /// Gets the image Height of depth frame
        /// </summary>
        public int DepthHeight
        {
            get
            {
                return this.depthHeight;
            }
        }

        /// <summary>
        /// Gets the image Width of color frame
        /// </summary>
        public int ColorWidth
        {
            get
            {
                return this.colorWidth;
            }
        }

        /// <summary>
        /// Gets the image Height of color frame
        /// </summary>
        public int ColorHeight
        {
            get
            {
                return this.colorHeight;
            }
        }

        /// <summary>
        /// Gets the intermediate storage for the extended depth data received from the camera in the current frame
        /// </summary>
        public DepthImagePixel[] DepthImagePixels
        {
            get
            {
                return this.depthImagePixels;
            }
        }

        /// <summary>
        /// Gets the intermediate storage for the color data received from the camera in 32bit color
        /// </summary>
        public byte[] ColorImagePixels
        {
            get
            {
                return this.colorImagePixels;
            }
        }

        /// <summary>
        /// Gets a value indicating whether this Kinect is used in reconstruction
        /// </summary>
        public bool UseSensor
        {
            get
            {
                if (this.reconstructionSensorControl != null)
                {
                    return this.reconstructionSensorControl.UseSensor;
                }
                else
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Gets a value indicating whether the Kinect sensor is using near mode
        /// </summary>
        public bool NearMode
        {
            get
            {
                if (this.reconstructionSensorControl != null)
                {
                    return this.reconstructionSensorControl.NearMode;
                }
                else
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Gets a value indicating whether the Kinect sensor is mirroring depth
        /// </summary>
        public bool MirrorDepth
        {
            get
            {
                if (this.reconstructionSensorControl != null)
                {
                    return this.reconstructionSensorControl.MirrorDepth;
                }
                else
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Gets a value indicating whether the reconstruction should integrate color from this camera
        /// </summary>
        public bool CaptureColor
        {
            get
            {
                if (this.reconstructionSensorControl != null)
                {
                    return this.reconstructionSensorControl.CaptureColor;
                }
                else
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Gets the min depth clip
        /// </summary>
        public float MinDepthClip
        {
            get
            {
                if (this.reconstructionSensorControl != null)
                {
                    return (float)this.reconstructionSensorControl.MinDepthClip;
                }
                else
                {
                    return FusionDepthProcessor.DefaultMinimumDepth;
                }
            }
        }

        /// <summary>
        /// Gets the max depth clip
        /// </summary>
        public float MaxDepthClip
        {
            get
            {
                if (this.reconstructionSensorControl != null)
                {
                    return (float)this.reconstructionSensorControl.MaxDepthClip;
                }
                else
                {
                    return FusionDepthProcessor.DefaultMaximumDepth;
                }
            }
        }

        #endregion

        /// <summary>
        /// Dispose resources
        /// </summary>
        public void Dispose()
        {
            this.Dispose(true);

            // This object will be cleaned up by the Dispose method.
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Start depth stream at specific resolution
        /// </summary>
        /// <param name="format">The resolution of image in depth stream</param>
        /// <returns>Returns true if the sensor supports near mode.</returns>
        public bool StartDepthStream(DepthImageFormat format)
        {
            if (null == this.sensor)
            {
                this.StatusMessage = "No ready Kinect found!";
                return true;
            }

            bool isSupportNearMode = true;

            try
            {
                // Enable depth stream, register event handler and start
                this.Sensor.DepthStream.Enable(format);
                this.Sensor.DepthFrameReady += this.OnDepthFrameReady;
                this.depthFormat = format;

                if (!this.IsStarted())
                {
                    this.Sensor.Start();
                    this.started = true;
                }

                // Set Near Mode by default
                try
                {
                    this.Sensor.DepthStream.Range = DepthRange.Near;
                    this.ReconSensorControl.NearMode = true;
                }
                catch (InvalidOperationException)
                {
                    isSupportNearMode = false;
                }

                // Create frustum and graphics camera
                Size imageSize = Helper.GetImageSize(this.DepthFormat);
                this.depthWidth = (int)imageSize.Width;
                this.depthHeight = (int)imageSize.Height;

                // Create the graphics camera and set at origin initially - we will override by setting the transform explicitly below
                this.reconstructionSensorCamera = new GraphicsCamera(new Point3D(0, 0, 0), Quaternion.Identity, (float)this.depthWidth / (float)this.depthHeight);

                // Update view transform now, from ReconstructionSensorControl
                this.SetCameraTransformation((float)this.reconstructionSensorControl.AngleX, (float)this.reconstructionSensorControl.AngleY, (float)this.reconstructionSensorControl.AngleZ, (float)this.reconstructionSensorControl.AxisDistance);
            }
            catch (IOException ex)
            {
                // Device is in use
                this.sensor = null;
                this.StatusMessage = ex.Message;
                throw;
            }
            catch (InvalidOperationException ex)
            {
                // Device is not valid, not supported or hardware feature unavailable
                this.sensor = null;
                this.StatusMessage = ex.Message;
                throw;
            }

            this.StatusMessage = string.Empty;

            return isSupportNearMode;
        }

        /// <summary>
        /// Stop depth stream and sensor
        /// </summary>
        public void StopDepthStream()
        {
            if (null != this.Sensor)
            {
                this.Sensor.DepthFrameReady -= this.OnDepthFrameReady;

                // this.Sensor.Stop();   // Unommenting this causes freeze on exit
                this.started = false;
            }
        }

        /// <summary>
        /// Start color stream at specific resolution
        /// </summary>
        /// <param name="format">The resolution of image in color stream</param>
        public void StartColorStream(ColorImageFormat format)
        {
            if (null == this.sensor)
            {
                this.StatusMessage = "No ready Kinect found!";
                return;
            }

            try
            {
                // Enable color stream, register event handler and start
                if (!this.Sensor.ColorStream.IsEnabled)
                {
                    this.Sensor.ColorStream.Enable(format);
                }

                this.Sensor.ColorFrameReady += this.OnColorFrameReady;
                this.colorFormat = format;

                if (!this.Sensor.IsRunning)
                {
                    this.Sensor.Start();
                    this.started = true;
                }
            }
            catch (IOException ex)
            {
                // Device is in use
                this.sensor = null;
                this.StatusMessage = ex.Message;
                throw;
            }
            catch (InvalidOperationException ex)
            {
                // Device is not valid, not supported or hardware feature unavailable
                this.sensor = null;
                this.StatusMessage = ex.Message;
                throw;
            }

            this.StatusMessage = string.Empty;
        }

        /// <summary>
        /// Stop color stream by disabling, rather than stopping sensor
        /// </summary>
        public void StopColorStream()
        {
            if (null != this.Sensor)
            {
                this.Sensor.ColorStream.Disable();
                this.Sensor.ColorFrameReady -= this.OnColorFrameReady;
            }
        }

        /// <summary>
        /// Returns true if the camera has been started
        /// </summary>
        /// <returns>Returns true if the camera is started.</returns>
        public bool IsStarted()
        {
            return this.started;
        }

        /// <summary>
        /// Check and enable or disable near mode
        /// </summary>
        public void CheckNearMode()
        {
            if (null != this.sensor)
            {
                try
                {
                    this.sensor.DepthStream.Range = this.ReconSensorControl.NearMode ? DepthRange.Near : DepthRange.Default;
                }
                catch (InvalidOperationException)
                {
                    // Fail silently
                    this.ReconSensorControl.NearMode = false;
                }
            }
        }

        /// <summary>
        /// Force an update of the camera transformation from the UI settings
        /// </summary>
        public void UpdateCameraTransformation()
        {
            this.reconstructionSensorControl.SetCameraTransformation();
        }

        /// <summary>
        /// Force an update of capture color setting
        /// </summary>
        /// <param name="captureColor">Setting value for capture color option</param>
        public void UpdateCaptureColor(bool captureColor)
        {
            this.reconstructionSensorControl.UpdateCaptureColor(captureColor);
        }

        /// <summary>
        /// Load the Reconstruction Sensor settings from an .xml file
        /// </summary>
        /// <returns>Returns true if successful, false otherwise.</returns>
        public bool LoadSettings()
        {
            bool successfulLoad = false;
            FileStream myFileStream = null;

            // Try to Load settings
            try
            {
                string sensorName = this.ParseKinectId();

                if (!string.IsNullOrEmpty(sensorName))
                {
                    XmlSerializer mySerializer = new XmlSerializer(typeof(ReconstructionSensorSettings));
                    myFileStream = new FileStream(sensorName + ".xml", FileMode.Open);
                    ReconstructionSensorSettings reconSensorSettings = (ReconstructionSensorSettings)mySerializer.Deserialize(myFileStream);

                    if (reconSensorSettings != null)
                    {
                        // Set these manually to update the UI
                        // This will then update the transformation from the new UI settings
                        this.reconstructionSensorControl.UpdateSettings(
                            reconSensorSettings.UseSensor,
                            reconSensorSettings.NearMode,
                            reconSensorSettings.MirrorDepth,
                            reconSensorSettings.CaptureColor,
                            reconSensorSettings.MinDepthClip,
                            reconSensorSettings.MaxDepthClip,
                            reconSensorSettings.AngleX,
                            reconSensorSettings.AngleY,
                            reconSensorSettings.AngleZ,
                            reconSensorSettings.AxisDistance);

                        successfulLoad = true;
                    }
                }
            }
            catch (ArgumentException)
            {
                // Fail silently
            }
            catch (FileNotFoundException)
            {
                // Fail silently
            }
            catch (DirectoryNotFoundException)
            {
                // Fail silently
            }
            catch (PathTooLongException)
            {
                // Fail silently
            }
            catch (UnauthorizedAccessException)
            {
                // Fail silently
            }
            catch (IOException)
            {
                // Fail silently
            }
            finally
            {
                if (null != myFileStream)
                {
                    // This will close the settings file, if open.
                    myFileStream.Dispose();
                }
            }

            return successfulLoad;
        }

        /// <summary>
        /// Save the Reconstruction Sensor settings to an .xml file
        /// </summary>
        /// <returns>Returns true if successful, false otherwise.</returns>
        public bool SaveSettings()
        {
            bool successfulSave = false;
            StreamWriter myWriter = null;

            // Try to Save settings
            try
            {
                string sensorName = this.ParseKinectId();

                if (!string.IsNullOrEmpty(sensorName))
                {
                    XmlSerializer mySerializer = new XmlSerializer(typeof(ReconstructionSensorSettings));
                    myWriter = new StreamWriter(sensorName + ".xml");

                    mySerializer.Serialize(myWriter, this.reconstructionSensorControl.SensorSettings);

                    successfulSave = true;
                }
            }
            catch (ArgumentException)
            {
                // Fail silently
            }
            catch (FileNotFoundException)
            {
                // Fail silently
            }
            catch (DirectoryNotFoundException)
            {
                // Fail silently
            }
            catch (PathTooLongException)
            {
                // Fail silently
            }
            catch (UnauthorizedAccessException)
            {
                // Fail silently
            }
            catch (IOException)
            {
                // Fail silently
            }
            finally
            {
                if (null != myWriter)
                {
                    // This will close the settings file, if open.
                    myWriter.Dispose();
                }
            }

            return successfulSave;
        }

        /// <summary>
        /// Mirror the color image in-place for display to match the depth image
        /// </summary>
        public unsafe void MirrorColorHorizontalInPlace()
        {
            if (null == ColorImagePixels)
            {
                return;
            }

            lock (this.colorLock)
            {
                // Here we make use of unsafe code to just copy the whole pixel as an int for performance reasons, as we do
                // not need access to the individual rgba components.
                fixed (byte* ptrColorPixels = ColorImagePixels)
                {
                    int* rawColorPixels = (int*)ptrColorPixels;

                    Parallel.For(
                        0,
                        colorHeight,
                        y =>
                        {
                            int index = y * colorWidth;
                            int mirrorIndex = index + colorWidth - 1;

                            for (int x = 0; x < (colorWidth / 2); ++x, ++index, --mirrorIndex)
                            {
                                // In-place swap to mirror
                                int temp = rawColorPixels[index];
                                rawColorPixels[index] = rawColorPixels[mirrorIndex];
                                rawColorPixels[mirrorIndex] = temp;
                            }
                        });
                }
            }
        }

        /// <summary>
        /// Process the color and depth inputs, converting the color into the depth space
        /// </summary>
        public unsafe void MapColorToDepth()
        {
            if (null == this.mapper)
            {
                // Create a coordinate mapper
                this.mapper = new CoordinateMapper(this.sensor);
            }

            this.mapper.MapDepthFrameToColorFrame(this.DepthFormat, this.depthImagePixels, this.ColorFormat, this.colorCoordinates);

            lock (this.colorLock)
            {
                if (this.MirrorDepth)
                {
                    // Here we make use of unsafe code to just copy the whole pixel as an int for performance reasons, as we do
                    // not need access to the individual rgba components.
                    fixed (byte* ptrColorPixels = this.colorImagePixels)
                    {
                        int* rawColorPixels = (int*)ptrColorPixels;

                        Parallel.For(
                            0,
                            this.depthHeight,
                            y =>
                            {
                                int destIndex = y * this.depthWidth;

                                for (int x = 0; x < this.depthWidth; ++x, ++destIndex)
                                {
                                    // calculate index into depth array
                                    int colorInDepthX = this.colorCoordinates[destIndex].X;
                                    int colorInDepthY = this.colorCoordinates[destIndex].Y;

                                    // make sure the depth pixel maps to a valid point in color space
                                    if (colorInDepthX >= 0 && colorInDepthX < this.colorWidth && colorInDepthY >= 0
                                        && colorInDepthY < this.colorHeight && this.depthImagePixels[destIndex].Depth != 0)
                                    {
                                        // Calculate index into color array
                                        int sourceColorIndex = colorInDepthX + (colorInDepthY * this.colorWidth);

                                        // Copy color pixel
                                        this.mappedColorPixels[destIndex] = rawColorPixels[sourceColorIndex];
                                    }
                                    else
                                    {
                                        this.mappedColorPixels[destIndex] = 0;
                                    }
                                }
                            });
                    }
                }
                else
                {
                    // Here we make use of unsafe code to just copy the whole pixel as an int for performance reasons, as we do
                    // not need access to the individual rgba components.
                    fixed (byte* ptrColorPixels = this.colorImagePixels)
                    {
                        int* rawColorPixels = (int*)ptrColorPixels;

                        Parallel.For(
                            0,
                            this.depthHeight,
                            y =>
                            {
                                int destIndex = y * this.depthWidth;
                                int flippedDestIndex = destIndex + (this.depthWidth - 1); // horizontally mirrored

                                for (int x = 0; x < this.depthWidth; ++x, ++destIndex, --flippedDestIndex)
                                {
                                    // calculate index into depth array, since the color is previously mirrored in place, we need to mirror the mapping result
                                    int colorInDepthX = this.colorWidth - this.colorCoordinates[destIndex].X;
                                    int colorInDepthY = this.colorCoordinates[destIndex].Y;

                                    // make sure the depth pixel maps to a valid point in color space
                                    if (colorInDepthX >= 0 && colorInDepthX < this.colorWidth && colorInDepthY >= 0
                                        && colorInDepthY < this.colorHeight && this.depthImagePixels[destIndex].Depth != 0)
                                    {
                                        // Calculate index into color array
                                        int sourceColorIndex = colorInDepthX + (colorInDepthY * this.colorWidth);

                                        // Copy color pixel, and re-mirror to destination
                                        this.mappedColorPixels[flippedDestIndex] = rawColorPixels[sourceColorIndex];
                                    }
                                    else
                                    {
                                        this.mappedColorPixels[flippedDestIndex] = 0;
                                    }
                                }
                            });
                    }
                }
            }

            this.MappedColorFrame.CopyPixelDataFrom(this.mappedColorPixels);
        }

        /// <summary>
        /// Frees all memory associated with the FusionImageFrame.
        /// </summary>
        /// <param name="disposing">Whether the function was called from Dispose.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!this.disposed)
            {
                if (disposing)
                {
                    if (null != this.reconstructionSensorControl)
                    {
                        this.reconstructionSensorControl.RequireResetReconstructionEvent -= this.OnResetReconstruction;
                        this.reconstructionSensorControl.SetTransformationEvent -= this.OnSetCameraTransformation;
                        this.reconstructionSensorControl.SetCaptureColorEvent -= this.OnSetCaptureColor;
                    }

                    this.depthImagePixels = null;
                    this.colorImagePixels = null;

                    if (null != this.DepthFloatFrame)
                    {
                        this.DepthFloatFrame.Dispose();
                    }

                    if (null != this.DeltaFromReferenceFrame)
                    {
                        this.DeltaFromReferenceFrame.Dispose();
                    }

                    if (null != this.ShadedSurfaceFrame)
                    {
                        this.ShadedSurfaceFrame.Dispose();
                    }

                    if (null != this.ShadedSurfaceNormalsFrame)
                    {
                        this.ShadedSurfaceNormalsFrame.Dispose();
                    }

                    if (null != this.PointCloudFrame)
                    {
                        this.PointCloudFrame.Dispose();
                    }

                    if (null != this.reconstructionSensorCamera)
                    {
                        this.reconstructionSensorCamera.Dispose();
                    }
                }
            }

            this.disposed = true;
        }

        /// <summary>
        /// Event handler for Kinect Sensor's DepthFrameReady event
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void OnDepthFrameReady(object sender, DepthImageFrameReadyEventArgs e)
        {
            bool savedFrame = false;

            // Open depth frame
            using (DepthImageFrame depthFrame = e.OpenDepthImageFrame())
            {
                if (null != depthFrame)
                {
                    this.StatusMessage = string.Empty;

                    // Save frame timestamp
                    this.FrameTimestamp = depthFrame.Timestamp;

                    // Create local depth pixels buffer
                    if (null == this.DepthImagePixels || this.DepthImagePixels.Length != depthFrame.PixelDataLength)
                    {
                        this.depthImagePixels = new DepthImagePixel[depthFrame.PixelDataLength];
                    }

                    // Copy depth pixels to local buffer
                    depthFrame.CopyDepthImagePixelDataTo(this.DepthImagePixels);

                    this.depthWidth = depthFrame.Width;
                    this.depthHeight = depthFrame.Height;

                    // Ensure frame resources are ready for Kinect Fusion
                    this.AllocateFrames();

                    savedFrame = true;
                }
            }

            // Signal that the depth is ready for processing
            if (savedFrame && null != this.DepthFrameReady)
            {
                this.DepthFrameReady(this, null);
            }
        }

        /// <summary>
        /// Event handler for Kinect Sensor's ColorFrameReady event
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void OnColorFrameReady(object sender, ColorImageFrameReadyEventArgs e)
        {
            bool savedFrame = false;

            lock (this.colorLock)
            {
                // Open color frame
                using (ColorImageFrame colorFrame = e.OpenColorImageFrame())
                {
                    if (null != colorFrame)
                    {
                        // Create local color pixels buffer
                        if (null == this.ColorImagePixels || this.ColorImagePixels.Length != this.Sensor.ColorStream.FramePixelDataLength)
                        {
                            this.colorImagePixels = new byte[this.Sensor.ColorStream.FramePixelDataLength];
                        }

                        colorFrame.CopyPixelDataTo(this.ColorImagePixels);

                        this.colorWidth = colorFrame.Width;
                        this.colorHeight = colorFrame.Height;

                        savedFrame = true;
                    }
                }
            }

            // Signal that the color frame is ready for processing
            if (savedFrame && null != this.ColorFrameReady)
            {
                this.ColorFrameReady(this, null);
            }
        }

        /// <summary>
        /// Fire Reset Reconstruction Event
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        private void OnResetReconstruction(object sender, EventArgs e)
        {
            if (this.RequireResetEvent != null)
            {
                this.RequireResetEvent(this, null);
            }
        }

        /// <summary>
        /// Fire Render Reconstruction Event
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        private void OnRenderReconstruction(object sender, EventArgs e)
        {
            if (this.RequireRenderEvent != null)
            {
                this.RequireRenderEvent(this, null);
            }
        }

        /// <summary>
        /// Set a Camera Transformation Event from the UI
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        private void OnSetCameraTransformation(object sender, TransformEventArgs e)
        {
            this.SetCameraTransformation(e.AngleX, e.AngleY, e.AngleZ, e.AxisDistance);
        }

        /// <summary>
        /// Set capture color value Event from the UI
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        private void OnSetCaptureColor(object sender, EventArgs e)
        {
            if (this.AllSetCaptureColorEvent != null)
            {
                this.AllSetCaptureColorEvent(this, null);
            }
        }

        /// <summary>
        /// Set a Camera Transformation
        /// Note: In WPF, the standard Right Hand coordinate system has the +X axis to the right, +Y axis up, and +Z axis out of the screen towards the viewer
        ///       ^ +Y
        ///       |
        ///       +----> +X
        ///      /
        ///     / +Z
        /// </summary>
        /// <param name="angleX">The rotation around the X axis.</param>
        /// <param name="angleY">The rotation around the Y axis.</param>
        /// <param name="angleZ">The rotation around the Z axis.</param>
        /// <param name="axisDistance">The distance from the origin.</param>
        private void SetCameraTransformation(float angleX, float angleY, float angleZ, float axisDistance)
        {
            Vector3D t = new Vector3D();
            t.Z = axisDistance; // along Z (which is correct if camera is situated along +Z, looking along Z towards origin

            // Axis-Aligned
            if (angleX == 0 && angleY == 0 && angleZ == 0)
            {
                Quaternion q = Quaternion.Identity;

                // Set identity pose with axis distance
                this.reconstructionSensorCamera.UpdateTransform(q, t);
            }
            else
            {
                Quaternion qx = new Quaternion(new Vector3D(1, 0, 0), angleX);
                Quaternion qy = new Quaternion(new Vector3D(0, 1, 0), angleY);
                Quaternion qz = new Quaternion(new Vector3D(0, 0, 1), angleZ);

                Quaternion q = qx * qy * qz;
                this.reconstructionSensorCamera.UpdateTransform(q, t);
            }

            // Raise a camera transform changed event
            if (null != this.SensorTransformationChanged)
            {
                this.SensorTransformationChanged(this, null);
            }
        }

        /// <summary>
        /// Convert the Kinect Camera device connection id string into a name we can save
        /// </summary>
        /// <returns>Returns the Kinect Camera device connection id string.</returns>
        private string ParseKinectId()
        {
            string[] splitString = this.DeviceConnectionId.Split('\\');

            if (splitString.Length > 0)
            {
                // Return last part of unique id, replacing any '&' characters with '_'
                string id = splitString[splitString.Length - 1].Replace('&', '_');
                return id;
            }
            else
            {
                return string.Empty;
            }
        }

        /// <summary>
        /// Allocate the frame buffers used in the process
        /// </summary>
        private void AllocateFrames()
        {
            // Allocate depth float frame
            if (null == this.DepthFloatFrame || this.DepthWidth != this.DepthFloatFrame.Width || this.DepthHeight != this.DepthFloatFrame.Height)
            {
                this.DepthFloatFrame = new FusionFloatImageFrame(this.DepthWidth, this.DepthHeight);
            }

            // Allocate delta from reference frame
            if (null == this.DeltaFromReferenceFrame || this.DepthWidth != this.DeltaFromReferenceFrame.Width || this.DepthHeight != this.DeltaFromReferenceFrame.Height)
            {
                this.DeltaFromReferenceFrame = new FusionFloatImageFrame(this.DepthWidth, this.DepthHeight);
            }

            // Allocate point cloud frame
            if (null == this.PointCloudFrame || this.DepthWidth != this.PointCloudFrame.Width || this.DepthHeight != this.PointCloudFrame.Height)
            {
                this.PointCloudFrame = new FusionPointCloudImageFrame(this.DepthWidth, this.DepthHeight);
            }

            // Allocate shaded surface frame
            if (null == this.ShadedSurfaceFrame || this.DepthWidth != this.ShadedSurfaceFrame.Width || this.DepthHeight != this.ShadedSurfaceFrame.Height)
            {
                this.ShadedSurfaceFrame = new FusionColorImageFrame(this.DepthWidth, this.DepthHeight);
            }

            // Allocate shaded surface normals frame
            if (null == this.ShadedSurfaceNormalsFrame || this.DepthWidth != this.ShadedSurfaceNormalsFrame.Width || this.DepthHeight != this.ShadedSurfaceNormalsFrame.Height)
            {
                this.ShadedSurfaceNormalsFrame = new FusionColorImageFrame(this.DepthWidth, this.DepthHeight);
            }

            // Allocate color image mapped into depth frame
            if (null == this.MappedColorFrame || this.DepthWidth != this.MappedColorFrame.Width || this.DepthHeight != this.MappedColorFrame.Height)
            {
                this.MappedColorFrame = new FusionColorImageFrame(this.DepthWidth, this.DepthHeight);
            }

            int depthImageSize = this.DepthWidth * this.DepthHeight;

            if (null == this.colorCoordinates || depthImageSize != this.colorCoordinates.Length)
            {
                // Allocate the depth-color mapping points
                this.colorCoordinates = new ColorImagePoint[depthImageSize];
            }

            if (null == this.mappedColorPixels || depthImageSize != this.mappedColorPixels.Length)
            {
                // Allocate mapped color points (i.e. color in depth frame of reference)
                this.mappedColorPixels = new int[depthImageSize];
            }
        }
    }
}
