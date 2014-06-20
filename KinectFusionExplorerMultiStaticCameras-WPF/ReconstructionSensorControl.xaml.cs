// -----------------------------------------------------------------------
// <copyright file="ReconstructionSensorControl.xaml.cs" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// -----------------------------------------------------------------------

namespace Microsoft.Samples.Kinect.KinectFusionExplorer
{
    using System;
    using System.ComponentModel;
    using System.Globalization;
    using System.Windows;
    using System.Windows.Controls;
    using System.Windows.Data;

    /// <summary>
    /// Interaction logic for the <see cref="ReconstructionSensorControl"/> class
    /// </summary>
    public partial class ReconstructionSensorControl : UserControl, INotifyPropertyChanged
    {
        #region Fields

        /// <summary>
        /// The sensor settings
        /// </summary>
        private ReconstructionSensorSettings reconSensorSettings = new ReconstructionSensorSettings();

        #endregion

        /// <summary>
        /// Initializes a new instance of the <see cref="ReconstructionSensorControl"/> class.
        /// </summary>
        public ReconstructionSensorControl()
        {
            this.InitializeComponent();
        }

        #region Properties
        
        /// <summary>
        /// Delegate for Event for reset required on parameter change
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        internal delegate void RequireResetReconstructionEventHandler(object sender, EventArgs e);

        /// <summary>
        /// Delegate for Event for render required on parameter change
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        internal delegate void RequireRenderReconstructionEventHandler(object sender, EventArgs e);

        /// <summary>
        /// Delegate for Event for setting transformation on changed variables
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        internal delegate void SetTransformationEventHandler(object sender, TransformEventArgs e);

        /// <summary>
        /// Delegate for Event for setting capture color value
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        internal delegate void SetCaptureColorEventHandler(object sender, EventArgs e);

        /// <summary>
        /// Property change event
        /// </summary>
        public event PropertyChangedEventHandler PropertyChanged;

        /// <summary>
        /// Event for reconstruction reset required on parameter change
        /// </summary>
        internal event RequireResetReconstructionEventHandler RequireResetReconstructionEvent;

        /// <summary>
        /// Event for reconstruction render required on parameter change
        /// </summary>
        internal event RequireRenderReconstructionEventHandler RequireRenderReconstructionEvent;

        /// <summary>
        /// Event for setting transformation on changed variables
        /// </summary>
        internal event SetTransformationEventHandler SetTransformationEvent;

        /// <summary>
        /// Event for setting capture color value
        /// </summary>
        internal event SetCaptureColorEventHandler SetCaptureColorEvent;

        /// <summary>
        /// Gets or sets a value indicating whether the sensor should be used in reconstruction
        /// </summary>
        public bool UseSensor
        {
            get
            {
                return this.reconSensorSettings.UseSensor;
            }

            set
            {
                this.reconSensorSettings.UseSensor = value;
                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("UseSensor"));
                }
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether the sensor has near mode enabled
        /// </summary>
        public bool NearMode
        {
            get
            {
                return this.reconSensorSettings.NearMode;
            }

            set
            {
                this.reconSensorSettings.NearMode = value;
                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("NearMode"));
                }
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether the depth is mirrored
        /// </summary>
        public bool MirrorDepth
        {
            get
            {
                return this.reconSensorSettings.MirrorDepth;
            }

            set
            {
                this.reconSensorSettings.MirrorDepth = value;
                this.ResetReconstruction();

                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("MirrorDepth"));
                }
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether color is captured
        /// </summary>
        public bool CaptureColor
        {
            get
            {
                return this.reconSensorSettings.CaptureColor;
            }

            set
            {
                this.UpdateCaptureColor(value);

                this.SetCaptureColor();
            }
        }

        /// <summary>
        /// Gets or sets the min clip depth value
        /// </summary>
        public double MinDepthClip
        {
            get
            {
                return this.reconSensorSettings.MinDepthClip;
            }

            set
            {
                this.reconSensorSettings.MinDepthClip = (float)value;
                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("MinDepthClip"));
                }
            }
        }

        /// <summary>
        /// Gets or sets the max clip depth value
        /// </summary>
        public double MaxDepthClip
        {
            get
            {
                return this.reconSensorSettings.MaxDepthClip;
            }

            set
            {
                this.reconSensorSettings.MaxDepthClip = (float)value;
                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("MaxDepthClip"));
                }
            }
        }

        /// <summary>
        /// Gets or sets the sensor rotation around X
        /// </summary>
        public double AngleX
        {
            get
            {
                return this.reconSensorSettings.AngleX;
            }

            set
            {
                this.reconSensorSettings.AngleX = (float)value;
                this.SetCameraTransformation();

                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("AngleX"));
                }
            }
        }

        /// <summary>
        /// Gets or sets the sensor rotation around Y
        /// </summary>
        public double AngleY
        {
            get
            {
                return this.reconSensorSettings.AngleY;
            }

            set
            {
                this.reconSensorSettings.AngleY = (float)value;
                this.SetCameraTransformation();

                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("AngleY"));
                }
            }
        }

        /// <summary>
        /// Gets or sets the sensor rotation around Z
        /// </summary>
        public double AngleZ
        {
            get
            {
                return this.reconSensorSettings.AngleZ;
            }

            set
            {
                this.reconSensorSettings.AngleZ = (float)value;
                this.SetCameraTransformation();

                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("AngleZ"));
                }
            }
        }

        /// <summary>
        /// Gets or sets the sensor distance from the origin
        /// </summary>
        public double AxisDistance
        {
            get
            {
                return this.reconSensorSettings.AxisDistance;
            }

            set
            {
                this.reconSensorSettings.AxisDistance = (float)value;
                this.SetCameraTransformation();

                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("AxisDistance"));
                }
            }
        }

        /// <summary>
        /// Gets the sensor settings
        /// </summary>
        public ReconstructionSensorSettings SensorSettings
        {
            get
            {
                return this.reconSensorSettings;
            }
        }

        #endregion

        /// <summary>
        /// Update the UI from settings
        /// </summary>
        /// <param name="useSensor">True if the sensor is used for reconstruction.</param>
        /// <param name="nearMode">True if the sensor is using near mode.</param>
        /// <param name="mirrorDepth">True if the sensor is mirroring the depth image.</param>
        /// <param name="captureColor">True if the sensor should capture a color image.</param>
        /// <param name="minDepthClip">The near distance to clip the depth image pixels at. Closer distances will be set to 0.</param>
        /// <param name="maxDepthClip">The far distance to clip the depth image pixels at. Further distances will be set to 1000.</param>
        /// <param name="angleX">Rotation angle around X axis.</param>
        /// <param name="angleY">Rotation angle around Y axis.</param>
        /// <param name="angleZ">Rotation angle around Z axis.</param>
        /// <param name="axisDistance">Distance from world origin.</param>
        public void UpdateSettings(
            bool useSensor,
            bool nearMode,
            bool mirrorDepth,
            bool captureColor,
            float minDepthClip,
            float maxDepthClip,
            float angleX,
            float angleY,
            float angleZ,
            float axisDistance)
        {
            this.UseSensor = useSensor;
            this.NearMode = nearMode;
            this.MirrorDepth = mirrorDepth;
            this.CaptureColor = captureColor;
            this.MinDepthClip = minDepthClip;
            this.MaxDepthClip = maxDepthClip;
            this.AngleX = angleX;
            this.AngleY = angleY;
            this.AngleZ = angleZ;
            this.AxisDistance = axisDistance;

            this.SetCameraTransformation();
        }

         /// <summary>
         /// Fire the SetCameraTransformation event
         /// </summary>
         public void SetCameraTransformation()
         {
             if (this.SetTransformationEvent != null)
             {
                 this.SetTransformationEvent(
                     this, 
                     new TransformEventArgs(
                     (float)this.reconSensorSettings.AngleX, 
                     (float)this.reconSensorSettings.AngleY,
                     (float)this.reconSensorSettings.AngleZ,
                     (float)this.reconSensorSettings.AxisDistance));
             }
         }

         /// <summary>
         /// Just update the capture color value not fire SetCaptureColor event
         /// </summary>
         /// <param name="captureColor">Setting value for capture color option</param>
         public void UpdateCaptureColor(bool captureColor)
         {
             this.reconSensorSettings.CaptureColor = captureColor;
             this.RenderReconstruction();

             if (null != this.PropertyChanged)
             {
                 this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("CaptureColor"));
             }
         }

         /// <summary>
         /// Fire the ResetReconstruction event
         /// </summary>
         private void ResetReconstruction()
         {
             // Fire reset event
             if (this.RequireResetReconstructionEvent != null)
             {
                 this.RequireResetReconstructionEvent(this, null);
             }
         }

         /// <summary>
         /// Fire the RenderReconstruction event
         /// </summary>
         private void RenderReconstruction()
         {
             // Fire reset event
             if (this.RequireRenderReconstructionEvent != null)
             {
                 this.RequireRenderReconstructionEvent(this, null);
             }
         }

         /// <summary>
         /// Fire the SetCaptureColor event
         /// </summary>
         private void SetCaptureColor()
         {
             if (this.SetCaptureColorEvent != null)
             {
                 this.SetCaptureColorEvent(this, null);
             }
         }

         /// <summary>
         /// Set the defaults
         /// </summary>
         /// <param name="sender">Object sending the event</param>
         /// <param name="e">Event arguments</param>
         private void ResetCameraDefaultsButtonClick(object sender, RoutedEventArgs e)
         {
             this.reconSensorSettings = new ReconstructionSensorSettings();
             this.UpdateSettings(
                 this.reconSensorSettings.UseSensor,
                 this.reconSensorSettings.NearMode,
                 this.reconSensorSettings.MirrorDepth,
                 this.reconSensorSettings.CaptureColor,
                 this.reconSensorSettings.MinDepthClip,
                 this.reconSensorSettings.MaxDepthClip,
                 this.reconSensorSettings.AngleX,
                 this.reconSensorSettings.AngleY,
                 this.reconSensorSettings.AngleZ,
                 this.reconSensorSettings.AxisDistance);
         }
    }
    
    /// <summary>
    /// The transform event arguments used to pass per-camera transform data from the UI
    /// </summary>
    public class TransformEventArgs : EventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TransformEventArgs"/> class.
        /// </summary>
        /// <param name="angleX">Rotation angle around X axis.</param>
        /// <param name="angleY">Rotation angle around Y axis.</param>
        /// <param name="angleZ">Rotation angle around Z axis.</param>
        /// <param name="axisDistance">Distance from world origin.</param>
        public TransformEventArgs(float angleX, float angleY, float angleZ, float axisDistance)
        {
            this.AngleX = angleX;
            this.AngleY = angleY;
            this.AngleZ = angleZ;
            this.AxisDistance = axisDistance;
        }

        /// <summary>
        /// Gets the angle of rotation around the +X axis
        /// </summary>
        public float AngleX { get; private set; }

        /// <summary>
        /// Gets the angle of rotation around the +Y axis
        /// </summary>
        public float AngleY { get; private set; }

        /// <summary>
        /// Gets the angle of rotation around the +Z axis
        /// </summary>
        public float AngleZ { get; private set; }

        /// <summary>
        /// Gets the distance of the camera from the origin
        /// </summary>
        public float AxisDistance { get; private set; }
    }

    /// <summary>
    /// Convert Float Value to UI text for angles
    /// </summary>
    public class FloatValueToIntTextConverter : IValueConverter
    {
        /// <summary>
        /// Convert float value to text
        /// </summary>
        /// <param name="value">The object to convert.</param>
        /// <param name="targetType">The target type.</param>
        /// <param name="parameter">The parameter object.</param>
        /// <param name="culture">The conversion globalization information.</param>
        /// <returns>Returns the value converted to text.</returns>
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return ((double)value).ToString("000.0", CultureInfo.CurrentCulture);
        }

        /// <summary>
        /// Convert text to float value, clamping +/- 180
        /// </summary>
        /// <param name="value">The object to convert.</param>
        /// <param name="targetType">The target type.</param>
        /// <param name="parameter">The parameter object.</param>
        /// <param name="culture">The conversion globalization information.</param>
        /// <returns>Returns the text converted to a value.</returns>
        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            float val = 0;
            float.TryParse(value as string, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint | NumberStyles.AllowLeadingWhite | NumberStyles.AllowTrailingWhite, CultureInfo.CurrentCulture, out val);

            val = val < -180.0f ? -180.0f : val;
            val = val > 180.0f ? 180.0f : val;

            return val;
        }
    }
}
